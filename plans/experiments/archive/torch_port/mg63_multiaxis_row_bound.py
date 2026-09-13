"""mg63 -- the multiaxis forward's row-mask bound: the remedy, the mechanism,
and how far the rule reaches.

WHAT MG62 FOUND.  The multiaxis forward projector's 3.3-fold penalty at the
non-dividing size is carried by ONE kernel argument: ``num_rows``, the detector
row count the kernel masks its row lanes against.  The wrapper rounds the row
STRIDE up to a multiple of 16 (``launch_rows``) but passes the REAL row count as
the mask bound, and Triton compiles a much slower kernel when it cannot prove
that bound divisible by 16.  mg62 showed it in both directions at the body, one
integer at a time, at a fixed view batch of 128 views:

    512x448x384   baseline (bound 448)     73.86 ms
                  bound 447               231.82 ms
    513x449x385   baseline (bound 449)    231.32 ms
                  bound 448                74.66 ms
                  bound 464                76.90 ms

Every other candidate was flat within a percent: the values band stride (449,
450 and 464 all read ~231 ms), num_channels, num_slices, num_pixels, and the
ragged one-view batch (2.4 ms out of 1003).

WHY CONE DOES NOT PAY IT.  The cone forward wrapper passes ``launch_rows`` as
its kernel's ``num_rows``, so its mask bound is always a multiple of 16 and its
extra row lanes are ordinary live lanes whose atomics land in output rows the
wrapper slices off.  The multiaxis forward instead masks at the real row count.
Cone reads 307.1 ms and 310.7 ms at the same two cells.

WHAT THIS RUN MEASURES.

  Part A  the remedy, end to end.  A wrapper identical to the shipped one except
          that the kernel's row bound and the grid's row extent both take
          ``launch_rows``.  Timed through sparse_forward_project at both cells
          and compared against the shipped path's values, which is the claim
          that has to hold: the extra row lanes write only into rows the
          wrapper slices off, so every real row must be unchanged.
  Part B  Triton's own report on the two compiled kernels -- registers, spills,
          shared memory, PTX size -- at the divisible bound and at the real one.
          A wall-clock ratio says there is a difference; this says what it is.
  Part C  the rule's shape.  The row bound swept over 448 to 464 at the
          dividing cell, holding everything else fixed, so the threshold is
          measured rather than assumed: divisibility by 16, by 8, or something
          else.
  Part D  the same ablation on the multiaxis BACK body, which pays 1.17 times
          at the non-dividing size.  Its row count is the sinogram's own row
          stride, so it cannot be padded the way the forward's can; this run
          only asks whether the residue is the same mechanism.

THIS RUN EDITS NO LIBRARY FILE.  Part A binds its wrapper into the projector's
body list for one measurement and puts the original back.
"""
import contextlib
import json
import os
import time

import numpy as np
import torch

# ── run parameters ───────────────────────────────────────────────────────────
FOCUS_CELLS = [(512, 448, 384), (513, 449, 385)]
ELEVATION_DEG = 25.0
INPUT_SEED = 0
WARMUP = 1
TRIALS = 5
ABLATION_VIEW_BATCH = 128
# Part C's sweep of the row-mask bound at the dividing cell.  448 is the real
# count; the rest step through the two divisibility classes that could set the
# threshold, so the reading names it instead of assuming it.
ROW_BOUND_SWEEP = [440, 441, 444, 446, 447, 448, 449, 450, 452, 456, 460, 464]
OUT = os.environ.get("MG63_OUT", "mg63_multiaxis_row_bound.json")


# ── helpers (mg62's, unchanged) ──────────────────────────────────────────────
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


def reset_peak():
    if not torch.cuda.is_available():
        return 0
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


def peak_mb(resident_bytes=0):
    if not torch.cuda.is_available():
        return None, None
    peak = torch.cuda.max_memory_allocated()
    return peak / 2**20, (peak - resident_bytes) / 2**20


def fmt_mb(value):
    return "n/a" if value is None else f"{value:.1f}"


def rel_max_err(a, b):
    a = a.detach().to(torch.float64)
    b = b.detach().to(torch.float64)
    return float((a - b).abs().max()) / max(float(b.abs().max()), 1e-30)


class Cell:
    """One multiaxis cell, built as the metrics harness builds it."""

    def __init__(self, size):
        import mbirtorch
        self.size = tuple(int(x) for x in size)
        n_views = self.size[0]
        angles = np.linspace(0, np.pi, n_views, endpoint=False)
        elevation = np.full(n_views, np.deg2rad(ELEVATION_DEG), dtype=np.float64)
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
        self.sinogram = self.model._shard_sinogram(rng.standard_normal(
            self.size, dtype=np.float32))
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


# ── Part A: the remedy ───────────────────────────────────────────────────────
def remedied_forward(row_bound=None):
    """The shipped multiaxis forward wrapper with ONE change: the kernel's row
    mask bound and the grid's row extent take the PADDED row count instead of
    the real one, which is the convention the cone forward already uses.

    Everything else is the shipped wrapper, including the geometry builders,
    which keep the REAL row count -- the slice-to-row map has to stay anchored
    on the detector the model actually has.

    Why the extra row lanes are safe.  Their gather index is clamped into the
    band they were handed and masked by that band, so no read leaves ``values``.
    Their ``k_center`` is clamped to the wrapper's own inert bounds, so the
    float-to-int conversion stays defined however far the row sits.  And their
    atomics land at row offsets below ``launch_rows``, which is the plane's own
    row stride, in rows the wrapper slices off before it returns.  Every real
    row keeps the mask it had, so every real value is the shipped one.

    ``row_bound`` overrides the bound for Part C's sweep; None uses the padded
    count, which is the remedy.
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
        if slice_radius > tm.MULTIAXIS_FWD_MAX_SLICE_RADIUS:
            raise RuntimeError("this probe does not cover the delegating case")

        num_views, num_pixels = n_p.shape
        band_len = int(values.shape[1])
        launch_rows = padded_kernel_width(int(num_rows_r))
        # THE ONE CHANGE: the bound the kernel masks against.  The default is
        # the padded row count; Part C sweeps it.
        bound = launch_rows if row_bound is None else int(row_bound)
        # The plane has to hold every row lane the bound makes live, so a
        # sweep arm above the real row count allocates a taller plane.  At
        # the remedy's own bound this is exactly launch_rows, so the remedy
        # allocates what the shipped wrapper allocates.
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
        launch_key = ('mg63_mafwd', values.device.index, int(psf_radius),
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

    forward.__name__ = "row_bound_padded_multiaxis_forward"
    forward._mbirtorch_no_compile = True
    return forward


def part_a(cells):
    from mbirtorch import triton_multiaxis as tm
    rows = []
    for cell in cells:
        shipped_out = cell.project()
        shipped = timed(cell.project)
        original = cell.projectors._fwd_body_per_dev[0]
        body = remedied_forward()
        body._view_batch_cost = original._view_batch_cost
        cell.projectors._fwd_body_per_dev[0] = body
        try:
            resident = reset_peak()
            remedied = timed(cell.project)
            peak, transient = peak_mb(resident)
            out = cell.project()
            diff = rel_max_err(out, shipped_out)
            del out
        finally:
            cell.projectors._fwd_body_per_dev[0] = original
        del shipped_out
        row = {"cell": cell.label(), "shipped": stats(shipped),
               "remedied": stats(remedied), "peak_mb": peak,
               "transient_mb": transient, "rel_max_err_vs_shipped": diff,
               "speedup": stats(shipped)["median_ms"] / stats(remedied)["median_ms"]}
        rows.append(row)
        print(f"  {row['cell']}: shipped {row['shipped']['median_ms']:.1f} ms, "
              f"row-bound padded {row['remedied']['median_ms']:.1f} ms "
              f"({row['speedup']:.2f}x), transient {fmt_mb(transient)} MB, "
              f"value diff {diff:.2e}", flush=True)
    return rows


# ── Part B: what Triton compiled ─────────────────────────────────────────────
def compiled_objects():
    """Every compiled variant of the multiaxis forward kernel this process
    holds.  Triton has moved the cache between versions, so the walk accepts a
    dict, a tuple, or an object with a ``cache`` attribute, and keeps anything
    that reports registers."""
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
    """The fields Triton exposes on one compiled kernel."""
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


def part_b(cell):
    """Compile the kernel at a divisible row bound and at the real one, on the
    same cell and the same inputs, then read what Triton made of each.

    The variant a call produced is identified by OBJECT IDENTITY against what
    the cache held before it, because the cache is a dict and its order says
    nothing about which entry is new."""
    args = cell.body_args()
    params = cell.projectors._view_params_per_dev[0]
    views = params[:ABLATION_VIEW_BATCH]
    known = {id(obj) for obj in compiled_objects()}
    rows = []
    for label, bound in (("bound 464 (divisible)", 464),
                         ("bound 449 (real)", 449),
                         ("bound 447 (real, odd)", 447),
                         ("bound 448 (divisible)", 448)):
        body = remedied_forward(row_bound=bound)
        out = body(cell.cylinders, cell.index_tensor, views, slice_start=0,
                   plan=None, **args)
        del out
        sync()
        fresh = [obj for obj in compiled_objects() if id(obj) not in known]
        known |= {id(obj) for obj in fresh}
        report = describe(fresh[0]) if fresh else None
        rows.append({"label": label, "bound": bound, "report": report,
                     "new_variants": len(fresh)})
        if report is None:
            print(f"  {label}: no new compiled variant (already cached)",
                  flush=True)
        else:
            print(f"  {label}: regs={report.get('n_regs')} "
                  f"spills={report.get('n_spills')} "
                  f"shared={report.get('shared')} "
                  f"ptx={report.get('ptx_size')} "
                  f"cubin={report.get('cubin_size')}", flush=True)
    return rows


# ── Part C: where the threshold sits ─────────────────────────────────────────
def part_c(cell):
    args = cell.body_args()
    params = cell.projectors._view_params_per_dev[0]
    views = params[:ABLATION_VIEW_BATCH]
    rows = []
    for bound in ROW_BOUND_SWEEP:
        body = remedied_forward(row_bound=bound)

        def run(body=body):
            return body(cell.cylinders, cell.index_tensor, views,
                        slice_start=0, plan=None, **args)

        times = timed(run)
        row = {"cell": cell.label(), "row_bound": bound,
               "divisible_by_16": bound % 16 == 0,
               "divisible_by_8": bound % 8 == 0, **stats(times)}
        rows.append(row)
        print(f"  bound {bound:4d}  {row['median_ms']:8.2f} ms   "
              f"div16={row['divisible_by_16']!s:5}  "
              f"div8={row['divisible_by_8']!s:5}", flush=True)
    return rows


# ── Part D: the back body's residue ──────────────────────────────────────────
def part_d(cells):
    """The multiaxis BACK body under the same one-integer ablation.  Its
    ``num_rows`` is the sinogram's own row stride, so it cannot be padded the
    way the forward's bound can; this only asks whether the 1.17-fold residue
    at the non-dividing size is the same mechanism."""
    rows = []
    for cell in cells:
        args = cell.body_args()
        body = cell.projectors._back_body_per_dev[0]
        params = cell.projectors._view_params_per_dev[0]
        batch = min(ABLATION_VIEW_BATCH, cell.size[0])
        views = params[:batch]
        real_rows = int(args['num_rows_r'])
        variants = [("baseline", real_rows)]
        for candidate in (real_rows - 1, real_rows + 1,
                          real_rows - (real_rows % 16),
                          real_rows + (-real_rows % 16)):
            if candidate not in [v for _, v in variants] and candidate > 0:
                variants.append((f"num_rows_r={candidate}", candidate))
        base = None
        for name, bound in variants:
            call_args = dict(args)
            call_args['num_rows_r'] = bound
            # The back kernel reads the sinogram at a row stride of num_rows,
            # so the array handed in must have exactly that many rows.  A
            # fresh contiguous block per bound keeps the read in bounds and
            # keeps the arm a pure timing probe.
            sino = torch.zeros((batch, bound, cell.size[2]),
                               dtype=torch.float32, device=cell.device)
            keep = min(bound, int(cell.sinogram.shape[1]))
            sino[:, :keep] = cell.sinogram[:batch, :keep]

            def run(call_args=call_args, sino=sino):
                return body(sino, cell.index_tensor, views, coeff_power=1,
                            slice_start=0, band_slices=None, plan=None,
                            **call_args)

            times = timed(run)
            row = {"cell": cell.label(), "variant": name, "row_bound": bound,
                   "divisible_by_16": bound % 16 == 0, **stats(times)}
            if base is None:
                base = row["median_ms"]
            row["ratio_to_baseline"] = row["median_ms"] / base
            rows.append(row)
            del sino
            print(f"  {row['cell']:>14} back {name:<20} "
                  f"{row['median_ms']:8.2f} ms  div16="
                  f"{row['divisible_by_16']!s:5}", flush=True)
    return rows


# ── the run ──────────────────────────────────────────────────────────────────
def witnesses():
    import mbirtorch
    from mbirtorch import kernel_availability
    row = {"torch": torch.__version__, "mbirtorch_file": mbirtorch.__file__,
           "cuda_available": torch.cuda.is_available()}
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

    cells = [Cell(size) for size in FOCUS_CELLS]
    for cell in cells:
        body = cell.projectors._fwd_body_per_dev[0]
        name = getattr(body, "__name__", str(body))
        if "triton" not in name:
            raise RuntimeError(f"{cell.label()} did not bind the Triton "
                               f"forward body (bound {name}); this probe "
                               f"measures that body.")

    print("\n### Part A -- the row-bound remedy, end to end")
    report["part_a"] = part_a(cells)

    print("\n### Part B -- what Triton compiled at each bound")
    report["part_b"] = part_b(cells[0])

    print("\n### Part C -- where the threshold sits")
    report["part_c"] = part_c(cells[0])

    print("\n### Part D -- the back body under the same ablation")
    report["part_d"] = part_d(cells)

    with open(OUT, "w") as sink:
        json.dump(report, sink, indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
