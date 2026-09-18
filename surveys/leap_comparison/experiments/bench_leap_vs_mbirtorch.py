"""Head-to-head timing of LEAP (LLNL) and mbirtorch on one NVIDIA H100.

The two libraries run in separate processes, chosen by --mode, so that their
CUDA runtimes never share an address space and so that each library's peak
memory reading covers only its own work.  Every measurement is appended to a
JSON-lines file as soon as it is taken, so a job that runs out of time still
leaves behind everything it finished.

Modes:
  phantom    write the shared phantom volume for one size
  mbirtorch  time mbirtorch's forward, back, FDK and iterative reconstruction
  leap       time LEAP's forward, back, FBP and iterative reconstruction
  compare    read the saved sinograms and reconstructions and cross-check them

Geometry (identical physical setup for both libraries):
  N views, N detector rows, N detector columns, an N x N x N volume.
  Source to rotation axis 1000 mm, source to detector 2000 mm, so the
  magnification is 2.  Detector pixel 2 * 256/N mm and voxel 1 * 256/N mm, so
  the reconstructed volume is always a 256 mm cube and the detector always
  covers it.  Angles are uniform over a full 360 degrees.
"""

import argparse
import json
import os
import subprocess
import threading
import time

import numpy as np

# Physical setup shared by both libraries.  Lengths in mm.
SOURCE_ISO_DIST = 1000.0
SOURCE_DETECTOR_DIST = 2000.0
VOLUME_SIDE_MM = 256.0          # the reconstructed cube is this wide at every N
BASE_N = 256                    # the size the pixel and voxel pitches are quoted at


def geometry(n):
    """Return the geometry for one problem size, in physical units."""
    delta_voxel = 1.0 * BASE_N / n           # mm per voxel
    delta_det = 2.0 * BASE_N / n             # mm per detector pixel, both directions
    return {
        'num_views': n, 'num_det_rows': n, 'num_det_channels': n,
        'recon_shape': (n, n, n),
        'delta_voxel_mm': delta_voxel,
        'delta_det_mm': delta_det,
        'source_iso_dist_mm': SOURCE_ISO_DIST,
        'source_detector_dist_mm': SOURCE_DETECTOR_DIST,
        'magnification': SOURCE_DETECTOR_DIST / SOURCE_ISO_DIST,
        'volume_side_mm': n * delta_voxel,
    }


# ── the shared phantom ───────────────────────────────────────────────────────
# The phantom is defined by physical positions and radii in mm, so the same
# object is built at every size and the sphere radii in voxels scale with N.
# One large sphere of value 1 holds two smaller spheres of value 0.5 and 2.
# The small spheres sit at deliberately unequal offsets in all three
# directions: an object with no accidental symmetry is what lets the
# alignment search below tell one candidate orientation from another.
BIG_SPHERE = ((0.0, 0.0, 0.0), 0.35 * VOLUME_SIDE_MM, 1.0)
SMALL_SPHERES = [((30.0, -20.0, 25.0), 25.0, 0.5),
                 ((-25.0, 35.0, -30.0), 18.0, 2.0)]


def phantom_path(results_dir, n):
    return os.path.join(results_dir, f'phantom_{n}.npy')


def make_phantom(n):
    """Build the phantom in mbirtorch's index order: (row = y, col = x, slice = z).

    mbirtorch maps its recon indices to physical coordinates as
        y = delta_voxel * (row   - (num_rows   - 1) / 2)
        x = delta_voxel * (col   - (num_cols   - 1) / 2)
        z = delta_voxel * (slice - (num_slices - 1) / 2)
    LEAP uses the same centered rule for x, y and z, so the same volume in
    LEAP's (z, y, x) order is just a transpose of this array; see
    phantom_for_leap below.
    """
    g = geometry(n)
    d = g['delta_voxel_mm']
    axis = d * (np.arange(n, dtype=np.float32) - (n - 1) / 2.0)
    yy = axis[:, None, None]
    xx = axis[None, :, None]
    zz = axis[None, None, :]
    volume = np.zeros((n, n, n), dtype=np.float32)
    (cx, cy, cz), radius, value = BIG_SPHERE
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= radius ** 2
    volume[inside] = value
    for (cx, cy, cz), radius, value in SMALL_SPHERES:
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= radius ** 2
        volume[inside] = value
    return volume


def phantom_for_leap(volume):
    """mbirtorch order (y, x, z) -> LEAP's default volume order (z, y, x)."""
    return np.ascontiguousarray(np.transpose(volume, (2, 0, 1)))


def leap_volume_to_mbirtorch(volume):
    """LEAP order (z, y, x) -> mbirtorch order (y, x, z)."""
    return np.ascontiguousarray(np.transpose(volume, (1, 2, 0)))


# ── measurement plumbing ─────────────────────────────────────────────────────
class GpuMemorySampler:
    """Sample the GPU's total used memory with nvidia-smi while work runs.

    LEAP allocates with cudaMalloc, outside any torch allocator, so the torch
    peak counter cannot see it.  This samples the whole device instead, which
    is a number both libraries can be compared on.  It includes the CUDA
    context and anything the torch caching allocator is holding, so it reads
    higher than the torch counter by design.
    """

    def __init__(self, uuids, interval=0.1):
        # uuids is the list of GPUs this process may use, in torch device
        # order, so a multi-GPU run reports one peak per device.  None means
        # take whatever nvidia-smi lists.
        self.uuids = uuids
        self.interval = interval
        self.max_by_uuid = {}
        self._stop = threading.Event()
        self._thread = None

    def _sample_once(self):
        try:
            out = subprocess.run(
                ['nvidia-smi', '--query-gpu=uuid,memory.used',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10).stdout
        except Exception:
            return
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 2:
                continue
            if self.uuids is not None and parts[0] not in self.uuids:
                continue
            try:
                value = float(parts[1])
            except ValueError:
                continue
            previous = self.max_by_uuid.get(parts[0])
            self.max_by_uuid[parts[0]] = value if previous is None else max(previous, value)

    def _run(self):
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval)

    def start(self):
        self.max_by_uuid = {}
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Return the per-device peaks, in the order the uuids were given."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self.uuids is None:
            return sorted(self.max_by_uuid.values(), reverse=True)
        return [self.max_by_uuid.get(u) for u in self.uuids]


def gpu_uuids():
    """The UUIDs of the visible GPUs, in torch device order.

    The sampler needs these so that it reads only the GPUs this job was given,
    and so that a multi-GPU run can report a peak for each of them separately.
    """
    try:
        import torch
        out = []
        for index in range(torch.cuda.device_count()):
            raw = getattr(torch.cuda.get_device_properties(index), 'uuid', None)
            if raw is None:
                return None
            text = str(raw)
            out.append(text if text.startswith('GPU-') else 'GPU-' + text)
        return out
    except Exception:
        return None


class Recorder:
    """Append one JSON object per measurement, flushing after each write."""

    def __init__(self, path, common):
        self.path = path
        self.common = common

    def write(self, **fields):
        record = dict(self.common)
        record.update(fields)
        with open(self.path, 'a') as handle:
            handle.write(json.dumps(record) + '\n')
        print('RECORD ' + json.dumps(record), flush=True)


def time_operation(run, sync, warmup, repeats, uuids):
    """Run an operation and return its best time and its memory readings.

    The torch peak counters are reset just before the timed repeats, and the
    caching allocator is emptied first so the whole-device sampler is not
    reading memory that an earlier measurement left cached.  Returns a dict
    that is written straight into the JSON record.  On a multi-GPU run the
    per-device lists are the interesting numbers; the two scalar fields are
    the largest single-device readings, so they stay comparable with the
    single-GPU records.
    """
    import torch
    device_count = torch.cuda.device_count()
    for _ in range(warmup):
        run()
        sync()
    for index in range(device_count):
        with torch.cuda.device(index):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    sampler = GpuMemorySampler(uuids)
    sampler.start()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        sync()
        times.append(time.perf_counter() - start)
    device_peaks = sampler.stop()
    torch_peaks = [torch.cuda.max_memory_allocated(index) / 2 ** 20
                   for index in range(device_count)]
    known = [v for v in device_peaks if v is not None]
    return {
        'time_s': min(times),
        'all_times_s': times,
        'torch_peak_mib': max(torch_peaks) if torch_peaks else None,
        'device_peak_mib': max(known) if known else None,
        'torch_peak_mib_per_gpu': torch_peaks,
        'device_peak_mib_per_gpu': device_peaks,
    }


def normalized_rmse(test, reference):
    """||test - reference|| / ||reference||, both flattened, in float64."""
    test = np.asarray(test, dtype=np.float64).ravel()
    reference = np.asarray(reference, dtype=np.float64).ravel()
    denominator = np.linalg.norm(reference)
    if denominator == 0.0:
        return float('nan')
    return float(np.linalg.norm(test - reference) / denominator)


# ── mbirtorch ────────────────────────────────────────────────────────────────
def run_mbirtorch(n, args):
    import torch
    import mbirtorch

    g = geometry(n)
    uuids = gpu_uuids()
    device = torch.device('cuda:0')
    device_count = torch.cuda.device_count()
    use_all = args.devices == 'all'

    def sync():
        for index in range(device_count if use_all else 1):
            torch.cuda.synchronize(index)

    recorder = Recorder(args.jsonl, {
        'library': 'mbirtorch',
        'library_version': getattr(mbirtorch, '__version__', 'unknown'),
        'torch_version': torch.__version__,
        'gpu_name': torch.cuda.get_device_name(0),
        'N': n,
        'devices_requested': args.devices,
        'visible_cuda_devices': device_count,
    })

    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float32)
    sinogram_shape = (g['num_views'], g['num_det_rows'], g['num_det_channels'])
    model = mbirtorch.ConeBeamModel(sinogram_shape, angles,
                                    source_detector_dist=g['source_detector_dist_mm'],
                                    source_iso_dist=g['source_iso_dist_mm'])
    model.set_params(delta_det_channel=g['delta_det_mm'],
                     delta_det_row=g['delta_det_mm'],
                     delta_voxel=g['delta_voxel_mm'],
                     recon_shape=g['recon_shape'])
    # 'one' pins a single device, 'all' pins every visible one, and 'auto'
    # leaves the layout to mbirtorch's own device policy, which is what a user
    # gets by default on a multi-GPU node.  Whichever is used, the layout the
    # model actually ran on is recorded after the first projection, because
    # the automatic policy settles at first use.
    if args.devices == 'one':
        model.configure_devices(num_devices=1)
    elif args.devices == 'all':
        model.configure_devices(num_devices=device_count)

    phantom = np.load(phantom_path(args.results_dir, n))
    phantom_gpu = torch.as_tensor(phantom, device=device)

    # Forward projection.
    sinogram_holder = {}

    def forward():
        sinogram_holder['value'] = model.forward_project(phantom_gpu, output_sharded=True)

    m = time_operation(forward, sync, args.warmup, args.repeats, uuids)
    recorder.write(operation='forward_project', repeats=args.repeats, **m)
    sinogram_gpu = sinogram_holder['value']

    try:
        placement_devices = [str(d) for d in model.recon_placement.devices]
    except Exception as error:
        placement_devices = [f'unavailable: {error}']
    recorder.write(operation='device_layout', devices=placement_devices,
                   num_devices=len(placement_devices))
    print('MBIRTORCH DEVICE LAYOUT ' + json.dumps(placement_devices), flush=True)

    if n == args.save_n:
        np.save(os.path.join(args.results_dir, f'sino_mbirtorch_{n}.npy'),
                sinogram_gpu.detach().cpu().numpy())

    # Back projection.
    def back():
        model.back_project(sinogram_gpu, output_sharded=True)

    m = time_operation(back, sync, args.warmup, args.repeats, uuids)
    recorder.write(operation='back_project', repeats=args.repeats, **m)

    # Direct (FDK) reconstruction.
    fdk_holder = {}

    def fdk():
        fdk_holder['value'] = model.recon_fdk(sinogram_gpu, output_sharded=True)

    m = time_operation(fdk, sync, args.warmup, args.repeats, uuids)
    recorder.write(operation='fdk', repeats=args.repeats, **m)

    if n == args.save_n:
        np.save(os.path.join(args.results_dir, f'fdk_mbirtorch_{n}.npy'),
                fdk_holder['value'].detach().cpu().numpy())

    # Adjoint check: <A x, y> against <x, A^T y> with random x and y.
    if n == args.save_n:
        generator = torch.Generator(device=device).manual_seed(0)
        x = torch.rand(g['recon_shape'], generator=generator, device=device,
                       dtype=torch.float32)
        y = torch.rand(sinogram_shape, generator=generator, device=device,
                       dtype=torch.float32)
        ax = model.forward_project(x, output_sharded=True)
        aty = model.back_project(y, output_sharded=True)
        left = float(torch.sum(ax.double() * y.double()).item())
        right = float(torch.sum(x.double() * aty.double()).item())
        relative = abs(left - right) / max(abs(left), abs(right))
        recorder.write(operation='adjoint_check', inner_forward=left,
                       inner_back=right, relative_difference=relative)
        del x, y, ax, aty
        torch.cuda.empty_cache()

    # Iterative reconstruction: exactly max_iterations VCD iterations, with
    # early stopping switched off by stop_threshold_change_pct = 0.
    iterative_repeats = args.iterative_repeats
    for run_index in range(iterative_repeats):
        def iterative():
            np.random.seed(0)
            model.recon(sinogram_gpu, max_iterations=args.iterations,
                        stop_threshold_change_pct=0.0, print_logs=False,
                        logfile_path=None, output_sharded=True)

        m = time_operation(iterative, sync, 0, 1, uuids)
        recorder.write(operation='iterative_recon',
                       time_per_iteration_s=m['time_s'] / args.iterations,
                       iterations=args.iterations, algorithm='VCD with qGGMRF prior',
                       run_index=run_index, repeats=1, **m)


# ── LEAP ─────────────────────────────────────────────────────────────────────
def run_leap(n, args):
    import torch
    import leapctype
    from leapctype import tomographicModels
    from leap_filter_sequence import filterSequence, TV

    g = geometry(n)
    uuids = gpu_uuids()
    device = torch.device('cuda:0')
    device_count = torch.cuda.device_count()
    use_all = args.devices == 'all'

    def sync():
        for index in range(device_count if use_all else 1):
            torch.cuda.synchronize(index)

    leapct = tomographicModels()
    if use_all:
        # LEAP spreads work over several GPUs only when the arrays live on the
        # HOST: leapctype passes "is this on the cpu" through to the library,
        # and data already sitting on one GPU is processed on that GPU alone.
        # So the multi-GPU arm below uses numpy arrays, and its times include
        # the host-to-device and device-to-host copies that buys.
        leapct.set_gpus(list(range(device_count)))
    else:
        leapct.set_gpu(0)
    gpus_in_use = [int(v) for v in np.asarray(leapct.get_gpus()).ravel()]

    recorder = Recorder(args.jsonl, {
        'library': 'leap',
        'library_version': str(leapct.version()),
        'torch_version': torch.__version__,
        'gpu_name': torch.cuda.get_device_name(0),
        'N': n,
        'devices_requested': args.devices,
        'visible_cuda_devices': device_count,
    })
    recorder.write(operation='device_layout', devices=gpus_in_use,
                   num_devices=len(gpus_in_use),
                   arrays_on='host (numpy)' if use_all else 'device (torch)')

    phis = np.ascontiguousarray(
        np.linspace(0.0, 360.0, n, endpoint=False), dtype=np.float32)
    leapct.set_conebeam(g['num_views'], g['num_det_rows'], g['num_det_channels'],
                        g['delta_det_mm'], g['delta_det_mm'],
                        0.5 * (g['num_det_rows'] - 1), 0.5 * (g['num_det_channels'] - 1),
                        phis, g['source_iso_dist_mm'], g['source_detector_dist_mm'])
    leapct.set_flatDetector()
    leapct.set_volume(n, n, n, voxelWidth=g['delta_voxel_mm'],
                      voxelHeight=g['delta_voxel_mm'])
    leapct.print_parameters()

    phantom = phantom_for_leap(np.load(phantom_path(args.results_dir, n)))
    projection_shape = (g['num_views'], g['num_det_rows'], g['num_det_channels'])
    if use_all:
        volume = np.ascontiguousarray(phantom, dtype=np.float32)
        projections = np.zeros(projection_shape, dtype=np.float32)
        scratch = np.zeros_like(volume)
    else:
        volume = torch.as_tensor(phantom, device=device).contiguous()
        projections = torch.zeros(projection_shape, dtype=torch.float32, device=device)
        scratch = torch.zeros_like(volume)

    def to_numpy(array):
        return array if isinstance(array, np.ndarray) else array.detach().cpu().numpy()

    # Forward projection.
    def forward():
        leapct.project(projections, volume)

    m = time_operation(forward, sync, args.warmup, args.repeats, uuids)
    recorder.write(operation='forward_project', repeats=args.repeats, **m)

    if n == args.save_n:
        np.save(os.path.join(args.results_dir, f'sino_leap_{n}.npy'),
                to_numpy(projections))

    # Back projection.  It writes into a scratch volume so the phantom is kept.
    def back():
        leapct.backproject(projections, scratch)

    m = time_operation(back, sync, args.warmup, args.repeats, uuids)
    recorder.write(operation='back_project', repeats=args.repeats, **m)

    # FBP, which is LEAP's FDK for cone-beam data.
    def fbp():
        leapct.FBP(projections, scratch)

    m = time_operation(fbp, sync, args.warmup, args.repeats, uuids)
    recorder.write(operation='fdk', repeats=args.repeats, **m)

    if n == args.save_n:
        np.save(os.path.join(args.results_dir, f'fdk_leap_{n}.npy'),
                to_numpy(scratch))

    # Adjoint check: <A x, y> against <x, A^T y> with random x and y.
    if n == args.save_n:
        generator = torch.Generator(device=device).manual_seed(0)
        x = torch.rand(volume.shape, generator=generator, device=device,
                       dtype=torch.float32).contiguous()
        y = torch.rand(projections.shape, generator=generator, device=device,
                       dtype=torch.float32).contiguous()
        ax = torch.zeros_like(projections)
        aty = torch.zeros_like(volume)
        leapct.project(ax, x)
        leapct.backproject(y, aty)
        torch.cuda.synchronize()
        left = float(torch.sum(ax.double() * y.double()).item())
        right = float(torch.sum(x.double() * aty.double()).item())
        relative = abs(left - right) / max(abs(left), abs(right))
        recorder.write(operation='adjoint_check', inner_forward=left,
                       inner_back=right, relative_difference=relative)
        del x, y, ax, aty
        torch.cuda.empty_cache()

    # Iterative reconstruction: FBP, then exactly args.iterations RWLS steps
    # with a TV regularizer.  The FBP start matches mbirtorch's recon, which
    # begins from its own direct reconstruction, so both timings cover
    # "sinogram in, N iterations out".  The weights are all ones, matching
    # mbirtorch's default of no measurement weighting.
    if use_all:
        weights = np.ones_like(projections)
        result = np.zeros_like(volume)
    else:
        weights = torch.ones_like(projections)
        result = torch.zeros_like(volume)

    for run_index in range(args.iterative_repeats):
        def iterative():
            filters = filterSequence(1.0)
            filters.append(TV(leapct, delta=0.025, p=1.2, weight=1.0))
            leapct.FBP(projections, result)
            leapct.RWLS(projections, result, args.iterations, filters=filters,
                        W=weights, preconditioner='SQS',
                        nonnegativityConstraint=True)

        m = time_operation(iterative, sync, 0, 1, uuids)
        recorder.write(operation='iterative_recon',
                       time_per_iteration_s=m['time_s'] / args.iterations,
                       iterations=args.iterations,
                       algorithm='RWLS with anisotropic TV, SQS preconditioner, FBP start',
                       run_index=run_index, repeats=1, **m)


# ── cross-checks ─────────────────────────────────────────────────────────────
def best_view_alignment(mbir_sino, leap_sino):
    """Find how LEAP's views and detector columns line up with mbirtorch's.

    The two libraries may number views from a different zero and may run the
    detector column index in the opposite direction.  Because the angles are
    uniform over a full turn, every candidate angle convention of the form
    phi -> +/- phi + constant is a shift, possibly with a reversal, of the view
    index.  So the search is over view shift, view reversal and column flip.
    It is done on cheap per-view column profiles (each view summed over
    detector rows) rather than on the full sinograms.
    """
    num_views = mbir_sino.shape[0]
    reference = mbir_sino.sum(axis=1).astype(np.float64)
    best = None
    for flip_columns in (False, True):
        candidate = leap_sino.sum(axis=1).astype(np.float64)
        if flip_columns:
            candidate = candidate[:, ::-1]
        for reverse_views in (False, True):
            ordered = candidate[::-1] if reverse_views else candidate
            for shift in range(num_views):
                rolled = np.roll(ordered, shift, axis=0)
                cost = float(np.linalg.norm(rolled - reference))
                if best is None or cost < best[0]:
                    best = (cost, flip_columns, reverse_views, shift)
    return {'profile_cost': best[0], 'flip_columns': best[1],
            'reverse_views': best[2], 'view_shift': best[3]}


def apply_alignment(leap_sino, alignment, flip_rows):
    out = leap_sino
    if alignment['flip_columns']:
        out = out[:, :, ::-1]
    if flip_rows:
        out = out[:, ::-1, :]
    if alignment['reverse_views']:
        out = out[::-1]
    out = np.roll(out, alignment['view_shift'], axis=0)
    return np.ascontiguousarray(out)


def run_compare(n, args):
    recorder = Recorder(args.jsonl, {'library': 'cross_check', 'N': n})
    phantom = np.load(phantom_path(args.results_dir, n))

    mbir_sino_path = os.path.join(args.results_dir, f'sino_mbirtorch_{n}.npy')
    leap_sino_path = os.path.join(args.results_dir, f'sino_leap_{n}.npy')
    if os.path.exists(mbir_sino_path) and os.path.exists(leap_sino_path):
        mbir_sino = np.load(mbir_sino_path)
        leap_sino = np.load(leap_sino_path)
        alignment = best_view_alignment(mbir_sino, leap_sino)
        best = None
        for flip_rows in (False, True):
            aligned = apply_alignment(leap_sino, alignment, flip_rows)
            raw = normalized_rmse(aligned, mbir_sino)
            # Also report the error after the single best global scale factor,
            # which separates a units or normalization difference from a
            # genuine disagreement about the geometry.
            a = aligned.astype(np.float64).ravel()
            b = mbir_sino.astype(np.float64).ravel()
            scale = float(np.dot(a, b) / np.dot(a, a)) if np.dot(a, a) > 0 else 1.0
            scaled = normalized_rmse(scale * aligned, mbir_sino)
            if best is None or scaled < best['nrmse_after_best_scale']:
                best = {'flip_rows': flip_rows, 'nrmse_raw': raw,
                        'nrmse_after_best_scale': scaled, 'best_scale': scale}
        recorder.write(operation='forward_projection_agreement',
                       alignment=dict(alignment, flip_rows=best['flip_rows']),
                       nrmse_raw=best['nrmse_raw'],
                       nrmse_after_best_scale=best['nrmse_after_best_scale'],
                       best_scale_leap_to_mbirtorch=best['best_scale'],
                       mbirtorch_sino_mean=float(mbir_sino.mean()),
                       leap_sino_mean=float(leap_sino.mean()))
    else:
        recorder.write(operation='forward_projection_agreement',
                       error='one or both saved sinograms are missing')

    # Each library's direct reconstruction against the phantom.  A direct
    # reconstruction is not expected to reproduce the phantom exactly, so this
    # number says the geometry is right, not that the algorithm is exact.
    for library, filename, to_mbirtorch_order in (
            ('mbirtorch', f'fdk_mbirtorch_{n}.npy', lambda v: v),
            ('leap', f'fdk_leap_{n}.npy', leap_volume_to_mbirtorch)):
        path = os.path.join(args.results_dir, filename)
        if not os.path.exists(path):
            recorder.write(operation='fdk_vs_phantom', which=library,
                           error='saved reconstruction is missing')
            continue
        volume = to_mbirtorch_order(np.load(path))
        options = {'as_is': volume, 'z_flipped': volume[:, :, ::-1]}
        results = {name: normalized_rmse(candidate, phantom)
                   for name, candidate in options.items()}
        chosen = min(results, key=results.get)
        candidate = options[chosen].astype(np.float64)
        reference = phantom.astype(np.float64)
        scale = float(np.dot(candidate.ravel(), reference.ravel())
                      / np.dot(candidate.ravel(), candidate.ravel()))
        recorder.write(operation='fdk_vs_phantom', which=library,
                       orientation=chosen, nrmse_raw=results[chosen],
                       nrmse_all_orientations=results,
                       nrmse_after_best_scale=normalized_rmse(scale * options[chosen],
                                                             phantom),
                       best_scale_recon_to_phantom=scale)


# ── entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True,
                        choices=['phantom', 'mbirtorch', 'leap', 'compare'])
    parser.add_argument('--N', type=int, required=True)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--jsonl', default=None)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--iterative-repeats', type=int, default=1)
    parser.add_argument('--save-n', type=int, default=256,
                        help='the size whose arrays are saved for the cross-checks')
    parser.add_argument('--devices', choices=['one', 'auto', 'all'], default='one',
                        help='one GPU (the default), every visible GPU, or, for '
                             'mbirtorch only, its own automatic device policy')
    args = parser.parse_args()
    if args.jsonl is None:
        args.jsonl = os.path.join(args.results_dir, 'raw_results.jsonl')
    os.makedirs(args.results_dir, exist_ok=True)

    if args.mode == 'phantom':
        path = phantom_path(args.results_dir, args.N)
        np.save(path, make_phantom(args.N))
        print(f'wrote {path}', flush=True)
        return
    if args.mode == 'mbirtorch':
        run_mbirtorch(args.N, args)
        return
    if args.mode == 'leap':
        run_leap(args.N, args)
        return
    run_compare(args.N, args)


if __name__ == '__main__':
    main()
