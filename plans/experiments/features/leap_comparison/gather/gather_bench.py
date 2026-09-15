"""Time the ways of assembling a slice-sharded volume on the host.

The volume lives on the devices as one contiguous shard per device, cut on
the last axis (the slices).  Assembling it on the host means writing each
shard into a strided block of one host array: every (row, column) pair
receives a run of that shard's slices.  The variants below differ in how
the shard bytes cross to the host and how they land in the strided block.

The variants, and the one thing each changes:

  library            Shards.gather() from LIBRARY_TREE, as it is.
  library_cap<K>, library_slot<M>MiB   the same with the library's thread
                     cap or slot size changed for the call, when the
                     library has those settings.
  threads_only       the library's per-shard copy, one thread per device.
  staged_seq_torch   pinned slab pipeline, devices one after another,
                     strided host write with torch's copy.
  staged_par_torch   the same pipeline with the devices concurrent.
  staged_par_numpy   the same, strided host write with a numpy assignment.
  staged_par_torch_<n>MiB   the same as staged_par_torch with another slot
                     size (the default slot is DEFAULT_SLOT_MIB).
  staged_par_numpy_t<T>, staged_par_torch_t<T>   the same pipelines with T
                     host threads per shard, each moving a range of the
                     block's rows through its own two slots (16 MiB).
  staged_par_torch_omp<K>   torch's intra-op thread count set to K for
                     the gather, so torch's own copy parallelizes the
                     strided write; _t4 adds four host threads per shard
                     on top, to see what oversubscription costs.
  staged_par_torch_prefault   the same as staged_par_torch with the host
                     array touched (zeroed) before the timer starts, which
                     removes the page faults from the figure.

Reference figures that bound the cost, not candidates for the library:

  transpose_layout   each shard transposed on its device to slice-major
                     order, so the host block is contiguous; the result is
                     in (slices, rows, columns) order.
  d2h_pinned_seq / d2h_pinned_par   each shard copied device to host into
                     a pinned buffer, nothing assembled: the transfer rate.
  d2h_pageable_seq / d2h_pageable_par   tensor.cpu() per shard: the
                     pageable transfer rate with the page faults of a new
                     host array.
  host_scatter_torch / host_scatter_numpy   the strided write alone, from
                     touched host copies of the shards into a new host
                     array, one thread per shard.

Each variant runs REPS times, on each count in DEVICE_COUNTS, and its
result is checked element for element against a plain concatenation on
its first run.  Run parameters are the constants below; the script takes
no arguments.  Results go to the log and to a JSON file in OUT_DIR.

With no CUDA device the script runs a small smoke pass on CPU tensors so
its logic can be checked on a laptop; the figures from that pass mean
nothing.
"""
import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor
import socket
import subprocess
import sys
import time

import numpy as np

# ---- run parameters -------------------------------------------------------
# The mbirtorch tree whose Shards.gather is timed as 'library'.  The batch
# file for a run against another tree sets GATHER_BENCH_LIBRARY_TREE.
LIBRARY_TREE = os.environ.get('GATHER_BENCH_LIBRARY_TREE',
                              '/scratch/gautschi/buzzard/leap_ornl/mbirtorch_src')
VOLUME_SHAPE = (1360, 1360, 1296)      # rows, columns, slices of the ORNL scan
DEVICE_COUNTS = [1, 2, 4]              # counts above the visible devices are skipped
REPS = 2
DEFAULT_SLOT_MIB = 64                  # one staging slot; two slots per device
SLOT_MIB_SWEEP = [16, 256]             # other slot sizes for staged_par_torch
THREADS_PER_SHARD_SWEEP = [1, 2, 4, 7, 14]   # host threads per shard
LIBRARY_CAP_SWEEP = [4, 8, 16, 32]     # the library's GATHER_THREADS_PER_SHARD
LIBRARY_SLOT_MIB_SWEEP = [8, 64]       # the library's GATHER_SLOT_BYTES, in MiB
OUT_DIR = '/scratch/gautschi/buzzard/gather_bench/out'
# ---------------------------------------------------------------------------

if os.path.isdir(LIBRARY_TREE):
    sys.path.insert(0, LIBRARY_TREE)
import torch                                              # noqa: E402
import mbirtorch                                          # noqa: E402
from mbirtorch import _sharding                           # noqa: E402
from mbirtorch._sharding import Placement, Shards, run_per_device   # noqa: E402

SMOKE = not torch.cuda.is_available()
if SMOKE:
    VOLUME_SHAPE = (37, 41, 53)
    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')


def log(*args):
    print(*args, flush=True)


def facts():
    log('host', socket.gethostname(), 'torch', torch.__version__,
        'numpy', np.__version__)
    log('library under test:', mbirtorch.__file__)
    if os.path.isdir(LIBRARY_TREE):
        assert mbirtorch.__file__.startswith(LIBRARY_TREE), mbirtorch.__file__
    cpus = (len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity')
            else os.cpu_count())
    log('cpus visible', cpus, 'torch threads', torch.get_num_threads())
    log('OMP_NUM_THREADS=%s MKL_NUM_THREADS=%s SLURM_CPUS_PER_TASK=%s'
        % tuple(os.environ.get(k) for k in
                ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'SLURM_CPUS_PER_TASK')))
    thp = '/sys/kernel/mm/transparent_hugepage/enabled'
    if os.path.exists(thp):
        with open(thp) as f:
            log('transparent hugepages:', f.read().strip())
    if SMOKE:
        log('SMOKE PASS on CPU tensors: the figures below mean nothing')
        return
    for i in range(torch.cuda.device_count()):
        log('cuda:%d' % i, torch.cuda.get_device_name(i))
    try:
        topo = subprocess.run(['nvidia-smi', 'topo', '-m'], capture_output=True,
                              text=True, timeout=60).stdout
        log(topo)
    except Exception as e:                                # noqa: BLE001
        log('nvidia-smi topo failed:', e)
    t0 = time.perf_counter()
    probe = torch.empty(DEFAULT_SLOT_MIB * 2 ** 18, dtype=torch.float32,
                        pin_memory=True)
    log('first pinned allocation of %d MiB: %.3f s'
        % (DEFAULT_SLOT_MIB, time.perf_counter() - t0))
    del probe


def all_devices():
    if SMOKE:
        return [torch.device('cpu')] * 4
    return [torch.device('cuda', i) for i in range(torch.cuda.device_count())]


def sync(devices):
    for d in devices:
        if d.type == 'cuda':
            torch.cuda.synchronize(d)


def make_shards(devices, shape, axis=-1):
    placement = Placement(devices, axis=axis, axis_len=shape[axis])
    tensors = []
    for k, (dev, (s, e)) in enumerate(placement.shard_ranges()):
        local = list(shape)
        local[axis] = e - s
        g = torch.Generator(device=dev).manual_seed(1000 + k)
        tensors.append(torch.rand(local, dtype=torch.float32, device=dev,
                                  generator=g))
    return Shards(tensors, placement)


def geometry(shards):
    """The whole array's shape and the block arithmetic shared by the
    variants: the shard axis, each shard's length and start on it, and the
    element counts before (outer) and after (inner) that axis."""
    first = shards.tensors[0]
    axis = shards.placement.axis % first.ndim
    lengths = [int(t.shape[axis]) for t in shards.tensors]
    shape = [int(n) for n in first.shape]
    shape[axis] = sum(lengths)
    outer = int(np.prod(shape[:axis])) if axis > 0 else 1
    inner = int(np.prod(shape[axis + 1:])) if axis + 1 < len(shape) else 1
    starts = [int(s) for s in np.cumsum([0] + lengths[:-1])]
    return axis, shape, lengths, starts, outer, inner


# ---- the staged (pinned slab) pipeline --------------------------------------
def slab_pieces(outer, width, slot_elems):
    """The (row0, row1, col0, col1) pieces of an (outer, width) grid, each of
    at most slot_elems elements.  Whole rows when a row fits in a slot,
    otherwise pieces of one row."""
    rows_per_slab = slot_elems // width
    if rows_per_slab >= 1:
        for r0 in range(0, outer, rows_per_slab):
            yield r0, min(r0 + rows_per_slab, outer), 0, width
    else:
        for r in range(outer):
            for c0 in range(0, width, slot_elems):
                yield r, r + 1, c0, min(c0 + slot_elems, width)


def copy_shard_through_slots(source, dest, slots, scatter):
    """Move one shard from its device into its strided host block one slab at
    a time, through two staging slots, so the copy of the next slab overlaps
    the host write of the current one.

    source: the shard as a contiguous 2-D device tensor (outer, width).
    dest:   the shard's block of the host array, a 2-D strided view of the
            same shape.
    slots:  two host tensors (pinned for a CUDA shard) of equal length.
    scatter: 'torch' writes the block with torch's copy, 'numpy' with a
            numpy assignment.
    """
    outer, width = source.shape
    slot_elems = slots[0].numel()
    pieces = list(slab_pieces(outer, width, slot_elems))
    on_cuda = source.device.type == 'cuda'
    stream = torch.cuda.current_stream(source.device) if on_cuda else None
    dest_np = dest.numpy() if scatter == 'numpy' else None

    def issue(k):
        r0, r1, c0, c1 = pieces[k]
        piece = source[r0:r1, c0:c1]
        view = slots[k % 2][:piece.numel()].view(r1 - r0, c1 - c0)
        view.copy_(piece, non_blocking=on_cuda)
        event = None
        if on_cuda:
            event = torch.cuda.Event()
            event.record(stream)
        return view, event

    if not pieces:
        return
    pending = issue(0)
    for k in range(len(pieces)):
        view, event = pending
        if k + 1 < len(pieces):
            pending = issue(k + 1)
        if event is not None:
            event.synchronize()
        r0, r1, c0, c1 = pieces[k]
        if scatter == 'numpy':
            dest_np[r0:r1, c0:c1] = view.numpy()
        else:
            dest[r0:r1, c0:c1].copy_(view)


def split_grid(rows, width, parts):
    """Up to ``parts`` (row0, row1, col0, col1) sub-grids of a (rows, width)
    grid: contiguous row ranges, or column ranges of the one row when the
    grid has a single row."""
    if rows == 1 and parts > 1:
        bounds = np.linspace(0, width, min(parts, width) + 1).astype(int)
        return [(0, 1, int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])
                if b > a]
    bounds = np.linspace(0, rows, min(parts, rows) + 1).astype(int)
    return [(int(a), int(b), 0, width) for a, b in zip(bounds[:-1], bounds[1:])
            if b > a]


def staged_gather(shards, slot_mib=DEFAULT_SLOT_MIB, concurrent=True,
                  scatter='torch', prefault=False, extra=None,
                  threads_per_shard=1, torch_threads=None):
    axis, shape, lengths, starts, outer, inner = geometry(shards)
    tensors = shards.tensors
    dtype = tensors[0].dtype
    out = torch.empty(shape, dtype=dtype)
    if prefault:
        t0 = time.perf_counter()
        out.zero_()
        if extra is not None:
            extra['prefault_seconds'] = time.perf_counter() - t0
    total = shape[axis]
    slot_elems = int(slot_mib * 2 ** 20) // tensors[0].element_size()
    out2d = out.view(outer, total * inner)

    def worker(i, dev):
        n = lengths[i]
        if n == 0:
            return
        pinned = dev.type == 'cuda'
        source = tensors[i].reshape(outer, n * inner)
        dest = out2d[:, starts[i] * inner:(starts[i] + n) * inner]

        def part(sub):
            r0, r1, c0, c1 = sub
            slots = [torch.empty(slot_elems, dtype=dtype, pin_memory=pinned)
                     for _ in range(2)]
            copy_shard_through_slots(source[r0:r1, c0:c1], dest[r0:r1, c0:c1],
                                     slots, scatter)

        subs = split_grid(outer, n * inner, threads_per_shard)
        if len(subs) == 1:
            part(subs[0])
        else:
            with ThreadPoolExecutor(max_workers=len(subs)) as pool:
                list(pool.map(part, subs))

    saved_threads = torch.get_num_threads()
    if torch_threads is not None:
        torch.set_num_threads(torch_threads)
    try:
        if concurrent:
            run_per_device(shards.placement.devices, worker)
        else:
            for i, dev in enumerate(shards.placement.devices):
                worker(i, dev)
    finally:
        torch.set_num_threads(saved_threads)
    return out.numpy()


# ---- the other variants and the references ----------------------------------
def library_gather(shards):
    return shards.gather()


def library_gather_with(shards, cap=None, slot_mib=None):
    """Shards.gather() with the library's thread cap or slot size changed for
    the call (only when the library has those settings)."""
    names = ('GATHER_THREADS_PER_SHARD', 'GATHER_SLOT_BYTES')
    saved = {k: getattr(_sharding, k) for k in names if hasattr(_sharding, k)}
    try:
        if cap is not None:
            _sharding.GATHER_THREADS_PER_SHARD = cap
        if slot_mib is not None:
            _sharding.GATHER_SLOT_BYTES = slot_mib * 2 ** 20
        return shards.gather()
    finally:
        for k, v in saved.items():
            setattr(_sharding, k, v)


def threads_only_gather(shards):
    axis, shape, lengths, starts, outer, inner = geometry(shards)
    out = torch.empty(shape, dtype=shards.dtype)

    def worker(i, dev):
        if lengths[i] == 0:
            return
        out.narrow(axis, starts[i], lengths[i]).copy_(
            shards.tensors[i].detach())

    run_per_device(shards.placement.devices, worker)
    return out.numpy()


def transpose_layout_gather(shards, slot_mib=DEFAULT_SLOT_MIB):
    """Approach 2: transpose each shard on its device to slice-major order
    and copy it into a contiguous block of a (slices, rows, columns) host
    array.  The result is not in the library's layout."""
    axis, shape, lengths, starts, outer, inner = geometry(shards)
    assert axis == len(shape) - 1
    tensors = shards.tensors
    dtype = tensors[0].dtype
    total = shape[axis]
    out = torch.empty([total] + shape[:-1], dtype=dtype)
    out2d = out.view(1, total * outer)
    slot_elems = int(slot_mib * 2 ** 20) // tensors[0].element_size()

    def worker(i, dev):
        n = lengths[i]
        if n == 0:
            return
        moved = tensors[i].movedim(-1, 0).contiguous()
        source = moved.reshape(1, n * outer)
        dest = out2d[:, starts[i] * outer:(starts[i] + n) * outer]
        slots = [torch.empty(slot_elems, dtype=dtype,
                             pin_memory=dev.type == 'cuda') for _ in range(2)]
        copy_shard_through_slots(source, dest, slots, 'torch')

    run_per_device(shards.placement.devices, worker)
    return out.numpy()


def d2h_pinned_ref(shards, concurrent, piece_mib=1024):
    """Each shard crosses to the host in pieces through one pinned buffer per
    device; nothing is assembled."""
    def worker(i, dev):
        t = shards.tensors[i].reshape(-1)
        if t.numel() == 0:
            return
        piece_elems = min(piece_mib * 2 ** 20 // t.element_size(), t.numel())
        buf = torch.empty(piece_elems, dtype=t.dtype, pin_memory=True)
        stream = torch.cuda.current_stream(dev)
        for a in range(0, t.numel(), piece_elems):
            b = min(a + piece_elems, t.numel())
            buf[:b - a].copy_(t[a:b], non_blocking=True)
        stream.synchronize()

    if concurrent:
        run_per_device(shards.placement.devices, worker)
    else:
        for i, dev in enumerate(shards.placement.devices):
            worker(i, dev)
    return None


def d2h_pageable_ref(shards, concurrent):
    """tensor.cpu() per shard: a contiguous pageable copy into a new host
    array, page faults included; nothing is assembled."""
    def worker(i, dev):
        return shards.tensors[i].cpu()

    if concurrent:
        got = run_per_device(shards.placement.devices, worker)
    else:
        got = [worker(i, dev) for i, dev in enumerate(shards.placement.devices)]
    del got
    return None


def host_scatter_ref(shards, host_pieces, scatter, threads_per_shard=1):
    """The strided write alone: touched host copies of the shards written
    into their blocks of a new host array, one or more threads per shard."""
    axis, shape, lengths, starts, outer, inner = geometry(shards)
    out = torch.empty(shape, dtype=shards.dtype)
    out2d = out.view(outer, shape[axis] * inner)

    def worker(i, dev):
        n = lengths[i]
        if n == 0:
            return
        src = host_pieces[i].view(outer, n * inner)
        dest = out2d[:, starts[i] * inner:(starts[i] + n) * inner]

        def part(sub):
            r0, r1, c0, c1 = sub
            if scatter == 'numpy':
                dest.numpy()[r0:r1, c0:c1] = src.numpy()[r0:r1, c0:c1]
            else:
                dest[r0:r1, c0:c1].copy_(src[r0:r1, c0:c1])

        subs = split_grid(outer, n * inner, threads_per_shard)
        if len(subs) == 1:
            part(subs[0])
        else:
            with ThreadPoolExecutor(max_workers=len(subs)) as pool:
                list(pool.map(part, subs))

    run_per_device(shards.placement.devices, worker)
    return out.numpy()


# ---- correctness of the library's gather on the real devices ----------------
def correctness_block(devices):
    """Shards.gather on the real devices against the host array they were
    cut from: both axes, uneven lengths, shards of zero length, two element
    sizes, and (when the library has a slot size) slots so small that a
    shard moves in many pieces and a row is split."""
    cases = 0
    rng = np.random.default_rng(7)
    slot_names = [n for n in ('GATHER_SLOT_BYTES',) if hasattr(_sharding, n)]
    slot_values = [None] + ([4096, 64] if slot_names else [])
    for n in DEVICE_COUNTS:
        if n > len(devices):
            continue
        for axis, shape in [(-1, (37, 41, 1001)), (0, (1001, 37, 41)),
                            (-1, (37, 41, 3)), (-1, (37, 41, 0)),
                            (-1, (5, 3, 2)), (1, (7, 1001, 6))]:
            for dtype in (np.float32, np.float64, np.uint8):
                host = rng.random(shape).astype(dtype) if dtype != np.uint8 \
                    else rng.integers(0, 255, shape).astype(dtype)
                p = Placement(devices[:n], axis=axis, axis_len=shape[axis])
                parts = []
                for dev, (s, e) in p.shard_ranges():
                    idx = [slice(None)] * len(shape)
                    idx[axis] = slice(s, e)
                    parts.append(torch.as_tensor(host[tuple(idx)]).to(dev))
                for slot in slot_values:
                    saved = {name: getattr(_sharding, name) for name in slot_names}
                    try:
                        if slot is not None:
                            for name in slot_names:
                                setattr(_sharding, name, slot)
                        got = Shards(parts, p).gather()
                    finally:
                        for name, value in saved.items():
                            setattr(_sharding, name, value)
                    assert got.shape == host.shape, (n, axis, shape, dtype, slot)
                    assert got.dtype == host.dtype, (n, axis, shape, dtype, slot)
                    assert got.flags['C_CONTIGUOUS'], (n, axis, shape, dtype, slot)
                    assert np.array_equal(got, host), (n, axis, shape, dtype, slot)
                    cases += 1
    log('correctness: Shards.gather matched the host array in %d cases '
        '(slot sizes tried: %s)' % (cases, slot_values))


# ---- the timing loop ---------------------------------------------------------
def main():
    facts()
    devices = all_devices()
    correctness_block(devices)
    results = []
    volume_bytes = int(np.prod(VOLUME_SHAPE)) * 4
    gib = volume_bytes / 2 ** 30

    for n in DEVICE_COUNTS:
        if n > len(devices):
            log('n=%d skipped: %d devices visible' % (n, len(devices)))
            continue
        subset = devices[:n]
        shards = make_shards(subset, VOLUME_SHAPE)
        sync(subset)
        ref = np.concatenate([t.cpu().numpy() for t in shards.tensors], axis=-1)
        host_pieces = [t.cpu() for t in shards.tensors]
        sync(subset)
        log('n=%d shards %s' % (n, [tuple(t.shape) for t in shards.tensors]))

        def same(res):
            return bool(np.array_equal(res, ref))

        def same_transposed(res):
            return bool(np.array_equal(res, np.moveaxis(ref, -1, 0)))

        cpus = (len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity')
                else os.cpu_count())
        per_shard = max(1, cpus // n)
        variants = [
            ('library', lambda: library_gather(shards), same),
        ]
        if hasattr(_sharding, 'GATHER_THREADS_PER_SHARD'):
            for cap in LIBRARY_CAP_SWEEP:
                variants.append(('library_cap%d' % cap,
                                 lambda cap=cap: library_gather_with(shards, cap=cap),
                                 same))
            for mib in LIBRARY_SLOT_MIB_SWEEP:
                variants.append(('library_slot%dMiB' % mib,
                                 lambda mib=mib: library_gather_with(
                                     shards, slot_mib=mib), same))
        variants += [
            ('threads_only', lambda: threads_only_gather(shards), same),
            ('staged_seq_torch',
             lambda: staged_gather(shards, concurrent=False), same),
            ('staged_par_torch', lambda: staged_gather(shards), same),
            ('staged_par_numpy',
             lambda: staged_gather(shards, scatter='numpy'), same),
        ]
        for mib in SLOT_MIB_SWEEP:
            variants.append(('staged_par_torch_%dMiB' % mib,
                             lambda mib=mib: staged_gather(shards, slot_mib=mib),
                             same))
        for t in THREADS_PER_SHARD_SWEEP:
            variants.append(('staged_par_numpy_t%d' % t,
                             lambda t=t: staged_gather(shards, scatter='numpy',
                                                       slot_mib=16,
                                                       threads_per_shard=t),
                             same))
        for t in THREADS_PER_SHARD_SWEEP[1::2]:
            variants.append(('staged_par_torch_t%d' % t,
                             lambda t=t: staged_gather(shards, slot_mib=16,
                                                       threads_per_shard=t),
                             same))
        for mib in [4, 64]:
            variants.append(('staged_par_numpy_t%d_%dMiB' % (per_shard, mib),
                             lambda mib=mib: staged_gather(
                                 shards, scatter='numpy', slot_mib=mib,
                                 threads_per_shard=per_shard), same))
        variants.append(('staged_par_torch_omp%d' % per_shard,
                         lambda: staged_gather(shards, torch_threads=per_shard),
                         same))
        variants.append(('staged_par_torch_omp%d_t4' % per_shard,
                         lambda: staged_gather(shards, torch_threads=per_shard,
                                               threads_per_shard=4), same))
        extra = {}
        variants.append(('staged_par_torch_prefault',
                         lambda: staged_gather(shards, prefault=True, extra=extra),
                         same))
        variants.append(('transpose_layout',
                         lambda: transpose_layout_gather(shards), same_transposed))
        if not SMOKE:
            variants += [
                ('d2h_pinned_seq', lambda: d2h_pinned_ref(shards, False), None),
                ('d2h_pinned_par', lambda: d2h_pinned_ref(shards, True), None),
                ('d2h_pageable_par', lambda: d2h_pageable_ref(shards, True), None),
            ]
        variants += [
            ('host_scatter_torch',
             lambda: host_scatter_ref(shards, host_pieces, 'torch'), same),
            ('host_scatter_numpy',
             lambda: host_scatter_ref(shards, host_pieces, 'numpy'), same),
        ]
        for t in THREADS_PER_SHARD_SWEEP[1:]:
            variants.append(('host_scatter_numpy_t%d' % t,
                             lambda t=t: host_scatter_ref(shards, host_pieces,
                                                          'numpy', t), same))

        for name, fn, check in variants:
            for rep in range(REPS):
                extra.clear()
                gc.collect()
                sync(subset)
                t0 = time.perf_counter()
                res = fn()
                sync(subset)
                seconds = time.perf_counter() - t0
                equal = check(res) if (check is not None and rep == 0) else None
                row = dict(n=n, variant=name, rep=rep, seconds=seconds, gib=gib,
                           gb_per_s=volume_bytes / 1e9 / seconds, equal=equal,
                           **extra)
                results.append(row)
                log('n=%d %-28s rep=%d %7.3f s  %6.2f GB/s  equal=%s %s'
                    % (n, name, rep, seconds, row['gb_per_s'], equal,
                       '' if not extra else extra))
                del res
                gc.collect()
        del shards, ref, host_pieces
        gc.collect()
        if not SMOKE:
            torch.cuda.empty_cache()

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = os.environ.get('SLURM_JOB_ID', 'local')
    path = os.path.join(OUT_DIR, 'gather_bench_%s.json' % tag)
    with open(path, 'w') as f:
        json.dump(dict(host=socket.gethostname(), library=mbirtorch.__file__,
                       volume_shape=list(VOLUME_SHAPE), results=results), f,
                  indent=1)
    log('results written to', path)
    log('GATHER BENCH DONE')


if __name__ == '__main__':
    main()
