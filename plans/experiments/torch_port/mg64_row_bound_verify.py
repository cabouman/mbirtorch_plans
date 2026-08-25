"""mg64 -- the row-bound change verified on the edited tree.

WHAT CHANGED.  ``_multiaxis_forward_view_batch_triton`` now passes the PADDED
detector row count as the kernel's row-mask bound and as the grid's row
extent, which is the convention the cone forward wrapper already used.  Before
the change it passed the real count, and mg62 and mg63 measured that as a
factor of 3.1 whenever the real count was not a multiple of 16.  The finding is
in multigpu_findings.md section 1.51.

WHAT THIS RUN CHECKS, in the order it runs.

  Part A  what Triton compiles at each bound: registers, spills, shared memory
          and code size, at a bound that is a multiple of 16 and at one that is
          not.  This runs FIRST because Triton compiles one kernel per
          divisibility class, so a later launch of either class would find the
          cache warm and report nothing.  This is the reading mg63's Part B
          could not take.
  Part B  the values against the TORCH BODY, which is the value reference, at
          every multiaxis cell the dashboard measures.  The edited kernel and
          the kernel's previous behaviour are both compared, so a gap the
          edited kernel shows can be told apart from a gap the kernel already
          had.  This kernel's gap grows with the detector row count, and the
          gate grows with it; see the two gates below.
  Part C  the same values against the kernel's OWN previous behaviour, built
          here by passing the real row count exactly as the shipped wrapper
          used to.  This isolates the edit rather than the whole kernel.
  Part D  the times, end to end through sparse_forward_project, at the four
          dashboard cells.  The two non-dividing cells should fall to about
          the dividing cells' cost and the dividing cells should not move.

The pytest run that gates correctness on real hardware is in the sbatch file
beside this script, not here: the CUDA-only kernel tests skip on any machine
without a GPU, so the cluster is the only place they run at all.
"""
import contextlib
import json
import os
import time

import numpy as np
import torch

# ── run parameters ───────────────────────────────────────────────────────────
CELLS = [(129, 113, 97), (256, 224, 192), (512, 448, 384), (513, 449, 385)]
# The cells Part A compares against the torch body.  The torch body materializes
# a (views, pixels, columns) gather, so the driver gives it a small view batch
# at the large cells and one comparison there costs seconds rather than
# milliseconds.  Every cell is still compared; the large ones are simply slow.
TORCH_REFERENCE_CELLS = CELLS
ELEVATION_DEG = 25.0
INPUT_SEED = 0
WARMUP = 1
TRIALS = 5
# The two gates, which are not the same number, and the reason they differ.
#
# Against the KERNEL'S PREVIOUS BEHAVIOUR, 1e-5: both sides round the row
# coordinate the same way, because they are the same kernel, so only the
# atomic summation order separates them.
#
# Against the TORCH BODY, the project's own two-tier rule, copied from
# tests/test_triton_multiaxis.py.  The trapezoid weight subtracts two row
# coordinates of size about num_rows_r, and the kernel and the torch body
# round the m0 + slope * k forming them differently, so the weight carries an
# absolute perturbation of about num_rows_r times float32 eps.  That is 5e-5
# at 448 rows, above a 1e-5 bar by itself.  The test file gates its small
# cells at 1e-5 and its multi-row-chunk cell at 1e-4, and splits them at the
# forward kernel's row tile; this run uses the same split.
VALUE_GATE_VS_PREVIOUS = 1e-5
VALUE_GATE_SMALL_ROWS = 1e-5
VALUE_GATE_MANY_ROWS = 1e-4
OUT = os.environ.get("MG64_OUT", "mg64_row_bound_verify.json")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, warmup=WARMUP, trials=TRIALS):
    for _ in range(warmup):
        out = fn()
        del out
    sync()
    times = []
    for _ in range(trials):
        sync()
        start = time.perf_counter()
        out = fn()
        sync()
        times.append((time.perf_counter() - start) * 1e3)
        del out
    return times


def stats(times):
    ordered = sorted(times)
    return {"min_ms": ordered[0], "median_ms": ordered[len(ordered) // 2],
            "max_ms": ordered[-1], "trials": len(ordered)}


def rel_max_err(a, b):
    a = a.detach().to(torch.float64)
    b = b.detach().to(torch.float64)
    return float((a - b).abs().max()) / max(float(b.abs().max()), 1e-30)


class Cell:
    """One multiaxis cell, built as the metrics harness builds it."""

    def __init__(self, size):
        import mbirtorch
        self.size = tuple(int(x) for x in size)
        angles = np.linspace(0, np.pi, self.size[0], endpoint=False)
        elevation = np.full(self.size[0], np.deg2rad(ELEVATION_DEG),
                            dtype=np.float64)
        self.model = mbirtorch.MultiAxisParallelModel(
            self.size, np.column_stack([angles, elevation]))
        reference = mbirtorch.ParallelBeamModel(self.size, angles)
        self.model.set_params(recon_shape=reference.get_params('recon_shape'),
                              no_warning=True)
        self.model.set_params(verbose=0, no_warning=True)
        self.model.configure_devices(num_devices=1)
        self.recon_shape = tuple(int(x) for x in
                                 self.model.get_params('recon_shape'))
        self.indices = mbirtorch.gen_full_indices(
            self.recon_shape, use_ror_mask=self.model.get_params('use_ror_mask'))
        self.num_pixels = int(len(self.indices))
        rng = np.random.default_rng(INPUT_SEED)
        self.cylinders = self.model._shard_recon(rng.standard_normal(
            (self.num_pixels, self.recon_shape[2]), dtype=np.float32))
        self.device = self.model.torch_device
        self.index_tensor = torch.as_tensor(self.indices, dtype=torch.int64,
                                            device=self.device)
        sync()

    @property
    def projectors(self):
        return self.model.projector_functions

    def body_args(self):
        return self.model._view_batch_args()

    def project(self):
        return self.model.sparse_forward_project(self.cylinders, self.indices)

    def label(self):
        return "x".join(str(x) for x in self.size)

    def rows_are_padded(self):
        from mbirtorch._utils import padded_kernel_width
        rows = int(self.body_args()['num_rows_r'])
        return padded_kernel_width(rows) != rows


def previous_forward(row_bound=None):
    """The forward wrapper as it stood BEFORE the change: the kernel's row
    bound and the grid's row extent take the REAL row count.

    Everything else is the shipped wrapper.  ``row_bound`` overrides the bound
    for Part D; None reproduces the previous behaviour.
    """
    from mbirtorch import triton_multiaxis as tm
    from mbirtorch._utils import padded_kernel_width
    from mbirtorch.multiaxis_parallel import (_multiaxis_horizontal_data,
                                              _multiaxis_vertical_terms)
    from mbirtorch.projectors import compile_serialized

    @torch.compiler.disable
    def forward(values, pixel_indices, view_params_batch, num_rows_r,
                num_channels, num_recon_rows, num_recon_cols, num_slices,
                delta_voxel, delta_voxel_row, delta_voxel_slice,
                delta_det_channel, delta_det_row, det_channel_offset,
                det_row_offset, recon_slice_offset, psf_radius,
                slice_start=0, plan=None):
        azimuth = view_params_batch[:, 0]
        elevation = view_params_batch[:, 1]
        n_p, centers, w_p_c, weight_scale, y = _multiaxis_horizontal_data(
            pixel_indices, azimuth, num_recon_rows, num_recon_cols,
            num_channels, delta_voxel, delta_voxel_row, delta_det_channel,
            det_channel_offset)
        m0, slope, w_p_r, l_max, scaling = _multiaxis_vertical_terms(
            y, azimuth, elevation, num_slices, delta_voxel, delta_voxel_row,
            delta_voxel_slice, delta_det_row, det_row_offset,
            recon_slice_offset, num_rows_r)
        slice_radius = tm._multiaxis_slice_tap_radius(w_p_r, slope)
        num_views, num_pixels = n_p.shape
        band_len = int(values.shape[1])
        launch_rows = padded_kernel_width(int(num_rows_r))
        bound = int(num_rows_r) if row_bound is None else int(row_bound)
        plane_rows = max(launch_rows, padded_kernel_width(bound))
        values = values.contiguous()
        contract = [t.contiguous() for t in (n_p, centers, m0)]
        contract += [t.reshape(num_views).contiguous()
                     for t in (w_p_c, weight_scale, slope, w_p_r, l_max,
                               scaling)]
        out = torch.zeros((num_views, num_channels, plane_rows),
                          dtype=torch.float32, device=values.device)
        block_p = tm._tile_size(tm.MULTIAXIS_FWD_BLOCK_P, num_pixels,
                                tm.MULTIAXIS_FWD_MIN_TILE)
        block_r = tm._tile_size(tm.MULTIAXIS_FWD_BLOCK_R, plane_rows,
                                tm.MULTIAXIS_FWD_MIN_TILE)
        grid = (-(-num_pixels // block_p), -(-bound // block_r), num_views)
        k_center_lo = float(int(slice_start) - slice_radius - 1)
        k_center_hi = float(int(slice_start) + band_len + slice_radius)
        launch_key = ('mg64_mafwd', values.device.index, int(psf_radius),
                      slice_radius, block_p, block_r, int(num_views),
                      int(num_pixels), int(num_channels), bound, plane_rows,
                      band_len, int(slice_start), int(num_slices))
        first = launch_key not in tm._COMPILED_LAUNCH_KEYS
        guard = compile_serialized() if first else contextlib.nullcontext()
        with torch.cuda.device(values.device), guard:
            tm._multiaxis_forward_kernel[grid](
                *contract, values, out,
                int(num_pixels), int(num_channels), bound,
                plane_rows, int(num_channels) * plane_rows,
                band_len, int(slice_start), int(num_slices),
                tm._SLOPE_FLOOR, k_center_lo, k_center_hi,
                PSF_RADIUS=int(psf_radius), SLICE_RADIUS=slice_radius,
                BLOCK_P=block_p, BLOCK_R=block_r,
                num_warps=tm.MULTIAXIS_FWD_NUM_WARPS,
                num_stages=tm.MULTIAXIS_FWD_NUM_STAGES)
        tm._COMPILED_LAUNCH_KEYS.add(launch_key)
        if plane_rows == int(num_rows_r):
            return out.permute(0, 2, 1)
        return out[:, :, :int(num_rows_r)].permute(0, 2, 1)

    forward.__name__ = "previous_multiaxis_forward"
    forward._mbirtorch_no_compile = True
    return forward


def one_batch(cell, body, views=None):
    """One forward body call over the cell's first view batch."""
    args = cell.body_args()
    params = cell.projectors._view_params_per_dev[0]
    count = views if views is not None else min(128, cell.size[0])
    return body(cell.cylinders, cell.index_tensor, params[:count],
                slice_start=0, plan=None, **args)


# ── Part B: the values against the torch body ────────────────────────────────
def torch_body_gate(cell):
    """The gate this cell's kernel-to-torch-body comparison is held to, and
    the row count that chooses it."""
    from mbirtorch import triton_multiaxis as tm
    rows = int(cell.body_args()['num_rows_r'])
    gate = (VALUE_GATE_MANY_ROWS if rows > tm.MULTIAXIS_FWD_BLOCK_R
            else VALUE_GATE_SMALL_ROWS)
    return gate, rows


def part_b(cells):
    """The edited kernel AND the kernel's previous behaviour, both against the
    torch body.  Two comparisons rather than one, because the second says
    whether any gap the first shows is one the edit introduced."""
    from mbirtorch.multiaxis_parallel import _multiaxis_forward_view_batch
    rows = []
    previous = previous_forward()
    for cell in cells:
        body = cell.projectors._fwd_body_per_dev[0]
        # A small view batch keeps the torch body's gather transient bounded at
        # the large cells; every side uses the same batch, so the comparison is
        # of the bodies and not of the batching.
        views = min(8, cell.size[0])
        gate, row_count = torch_body_gate(cell)
        reference = one_batch(cell, _multiaxis_forward_view_batch, views=views)
        kernel_out = one_batch(cell, body, views=views)
        diff = rel_max_err(kernel_out, reference)
        del kernel_out
        old_out = one_batch(cell, previous, views=views)
        old_diff = rel_max_err(old_out, reference)
        del old_out, reference
        row = {"cell": cell.label(), "views": views, "num_rows_r": row_count,
               "rows_are_padded": cell.rows_are_padded(), "gate": gate,
               "rel_max_err_vs_torch_body": diff,
               "previous_rel_max_err_vs_torch_body": old_diff,
               "passes": diff <= gate}
        rows.append(row)
        print(f"  {row['cell']:>14}  {row_count:4d} rows  gate {gate:.0e}  "
              f"edited {diff:.2e}  previous {old_diff:.2e}  "
              f"{'PASS' if row['passes'] else 'FAIL'}", flush=True)
    return rows


# ── Part C: the values against the kernel's previous behaviour ───────────────
def part_c(cells):
    rows = []
    previous = previous_forward()
    for cell in cells:
        body = cell.projectors._fwd_body_per_dev[0]
        views = min(128, cell.size[0])
        new_out = one_batch(cell, body, views=views)
        old_out = one_batch(cell, previous, views=views)
        diff = rel_max_err(new_out, old_out)
        row = {"cell": cell.label(), "views": views,
               "rows_are_padded": cell.rows_are_padded(),
               "rel_max_err_vs_previous": diff,
               "passes": diff <= VALUE_GATE_VS_PREVIOUS}
        del new_out, old_out
        rows.append(row)
        print(f"  {row['cell']:>14}  vs previous behaviour {diff:.2e}  "
              f"{'PASS' if row['passes'] else 'FAIL'}", flush=True)
    return rows


# ── Part D: the times ────────────────────────────────────────────────────────
def part_d(cells):
    rows = []
    for cell in cells:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            resident = torch.cuda.memory_allocated()
        times = timed(cell.project)
        peak = (torch.cuda.max_memory_allocated() if torch.cuda.is_available()
                else 0)
        row = {"cell": cell.label(),
               "rows_are_padded": cell.rows_are_padded(),
               "transient_mb": (peak - resident) / 2**20, **stats(times)}
        rows.append(row)
        print(f"  {row['cell']:>14}  {row['min_ms']:8.1f} ms min  "
              f"{row['median_ms']:8.1f} ms median  "
              f"{row['transient_mb']:7.1f} MB transient", flush=True)
    return rows


# ── Part A: what Triton compiled at each bound ───────────────────────────────
def compiled_objects():
    from mbirtorch import triton_multiaxis as tm
    kernel = tm._multiaxis_forward_kernel
    seen, found = set(), []

    def walk(node, depth=0):
        if depth > 4 or id(node) in seen:
            return
        seen.add(id(node))
        if hasattr(node, "n_regs") or hasattr(node, "metadata"):
            found.append(node)
            return
        if isinstance(node, dict):
            for value in node.values():
                walk(value, depth + 1)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value, depth + 1)
        else:
            for name in ("cache", "device_caches", "compiled"):
                child = getattr(node, name, None)
                if child is not None:
                    walk(child, depth + 1)

    for name in ("cache", "device_caches"):
        walk(getattr(kernel, name, None))
    return found


def describe(compiled):
    row = {}
    meta = getattr(compiled, "metadata", None)
    for field in ("name", "n_regs", "n_spills", "shared", "num_warps",
                  "num_stages"):
        value = getattr(compiled, field, None)
        if value is None and meta is not None:
            value = (meta.get(field) if isinstance(meta, dict)
                     else getattr(meta, field, None))
        row[field] = value
    asm = getattr(compiled, "asm", None)
    if isinstance(asm, dict):
        for form in ("ptx", "cubin"):
            if form in asm:
                row[form + "_size"] = len(asm[form])
    return row


def part_a(cell):
    """Both bounds compiled in one process, each identified by object identity
    against what the cache held before its launch.  The cell is chosen so that
    NEITHER bound has been compiled by the parts above."""
    known = {id(obj) for obj in compiled_objects()}
    rows = []
    for label, bound in (("bound 464, a multiple of 16", 464),
                         ("bound 449, not a multiple of 16", 449)):
        out = one_batch(cell, previous_forward(row_bound=bound), views=8)
        del out
        sync()
        fresh = [obj for obj in compiled_objects() if id(obj) not in known]
        known |= {id(obj) for obj in fresh}
        report = describe(fresh[0]) if fresh else None
        rows.append({"label": label, "bound": bound, "report": report})
        if report is None:
            print(f"  {label}: no new compiled variant", flush=True)
        else:
            print(f"  {label}: regs={report.get('n_regs')} "
                  f"spills={report.get('n_spills')} "
                  f"shared={report.get('shared')} "
                  f"ptx={report.get('ptx_size')} "
                  f"cubin={report.get('cubin_size')}", flush=True)
    return rows


# ── the run ──────────────────────────────────────────────────────────────────
def witnesses():
    import hashlib

    import mbirtorch
    from mbirtorch import kernel_availability, triton_multiaxis
    row = {"torch": torch.__version__, "mbirtorch_file": mbirtorch.__file__,
           "cuda_available": torch.cuda.is_available()}
    with open(triton_multiaxis.__file__, "rb") as source:
        row["triton_multiaxis_sha256"] = hashlib.sha256(
            source.read()).hexdigest()
    try:
        import triton
        row["triton"] = triton.__version__
    except Exception as error:                                   # noqa: BLE001
        row["triton"] = f"unavailable: {error}"
    if torch.cuda.is_available():
        row["device_name"] = torch.cuda.get_device_name(0)
    row["probe_usable"] = list(kernel_availability.triton_available())
    return row


def main():
    report = {"witnesses": witnesses()}
    print("### witnesses")
    for key, value in report["witnesses"].items():
        print(f"  {key}: {value}")

    cells = [Cell(size) for size in CELLS]
    for cell in cells:
        name = getattr(cell.projectors._fwd_body_per_dev[0], "__name__", "")
        if "triton" not in name:
            raise RuntimeError(f"{cell.label()} bound {name}, not the Triton "
                               f"forward body")

    # First, before any other launch warms Triton's cache for either class.
    print("\n### Part A -- what Triton compiled at each bound")
    report["part_a"] = part_a(cells[0])

    print("\n### Part B -- values against the torch body")
    report["part_b"] = part_b([c for c in cells
                               if tuple(c.size) in TORCH_REFERENCE_CELLS])

    print("\n### Part C -- values against the kernel's previous behaviour")
    report["part_c"] = part_c(cells)

    print("\n### Part D -- times end to end")
    report["part_d"] = part_d(cells)

    failed = [row for part in ("part_b", "part_c") for row in report[part]
              if not row["passes"]]
    with open(OUT, "w") as sink:
        json.dump(report, sink, indent=2)
    print(f"\nwrote {OUT}")
    if failed:
        print(f"VALUE GATE FAILED on {len(failed)} comparison(s)")
        return 1
    print("value gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
