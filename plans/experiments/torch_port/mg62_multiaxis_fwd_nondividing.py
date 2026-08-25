"""mg62 -- why the multiaxis forward projector costs 3.3 times more at the
non-dividing size than at the dividing one.

WHAT PROMPTED THIS RUN.  The nightly dashboard's first multiaxis rows, measured
2026-08-24 on one H100 at commit eee646af, read 306.7 ms for the forward
projector at the (512, 448, 384) cell and 1005.1 ms at (513, 449, 385).  Every
other row pays far less for the same step: the multiaxis filter pays 1.07 times
and its back projection 1.17 times, and parallel and cone pay 1.04 and 1.03
times on their own forward projectors.

WHAT THE RECORDED ROWS ALREADY SETTLE, so this run does not spend time on it.
The three warm trials at each cell agree to 0.2 ms, so the extra 700 ms is not
a compile landing inside a timed call.  Peak memory moves from 987.1 MB to
996.7 MB, one percent, so both cells run the Triton kernel: the torch body
would be charged the gather transient the driver prices itself, which forces a
much smaller view batch and a far larger peak.  And the (256, 224, 192) cell
reads 20.4 ms against the 512 cell's 306.7, a factor of 15.0 where the work
grows 16-fold, so the dividing cell sits on its own size curve and the
non-dividing cell is the outlier.

WHAT DIFFERS AT THE NON-DIVIDING SIZE.  Four things, and this run separates
them:

  band_len      the column count of ``values``, which is the SLICE GATHER'S
                ROW STRIDE inside the kernel.  It is num_slices, so 448 at the
                dividing cell and 449 at the non-dividing one.  It is the one
                width-class argument the multiaxis forward wrapper does not
                round up to a multiple of 16; the parallel forward wrapper
                rounds up exactly this argument (its ``launch_cols``) and pays
                1.04 times.
  num_rows_r    the detector row count, 448 or 449.  The wrapper rounds the
                row STRIDE up (``launch_rows``, 448 or 464) but passes the real
                count as the row mask bound.
  num_channels  384 or 385, a mask bound and a clamp bound.
  view batch    the driver's batch is 128 views for this body at both cells, so
                512 views split evenly into four batches while 513 views leave
                a ragged fifth batch of ONE view.

THE INSTRUMENT.  Five parts, cheapest first.

  Part A  reproduce the two dashboard numbers through the library's own funnel,
          sparse_forward_project, with the metrics harness's model and inputs.
          A run that does not reproduce them is measuring something else.
  Part B  split one projection into its body calls: how many, how long each,
          and how much of the call is not inside a body.
  Part C  read Triton's own report on the two compiled kernels -- registers,
          spills, shared memory -- which is authoritative where a wall-clock
          difference is not.
  Part D  the single-variable ablations, at the body wrapper, one integer
          changed at a time.  Two of them are VALUE-PRESERVING and run in both
          directions: padding the non-dividing cell's band up to 464 columns,
          and stretching the dividing cell's band to 449.  A zero column past
          num_slices is masked out of every tap, so those two arms change the
          stride and nothing else, and their outputs are compared.
  Part E  the candidate remedy end to end: bind a band-padding body and
          re-measure Part A, comparing values against the unpadded output.

THIS RUN EDITS NO LIBRARY FILE.  Part E binds its wrapper into the projector's
own body list for the duration of one measurement and puts the original back.
The exit code reports whether the instrument worked; the readings are read by a
person.
"""
import json
import os
import time

import numpy as np
import torch

# ── run parameters ───────────────────────────────────────────────────────────
# The two cells the dashboard measured, and the third one that shows the same
# inflation at a size small enough to be cheap.
CELLS = [(129, 113, 97), (256, 224, 192), (512, 448, 384), (513, 449, 385)]
# The cells Parts B, C, D and E work on: the pair the observation is about.
FOCUS_CELLS = [(512, 448, 384), (513, 449, 385)]
# The metrics harness's multiaxis geometry, unchanged: one elevation for every
# view, azimuths evenly spaced over half a turn, and the recon shape taken from
# a plain parallel model at the same sinogram size.
ELEVATION_DEG = 25.0
# The metrics harness's input seed for the forward op's cylinders.
INPUT_SEED = 0
# Timing: warm-ups are untimed, then the median and the minimum of the trials
# are reported.  The dashboard reports a minimum of three after one warm-up.
WARMUP = 1
TRIALS = 5
# The view batch one body-level ablation uses.  The driver chooses 128 for this
# body at both focus cells; the ablations hold that fixed so the only thing
# moving is the argument under test.
ABLATION_VIEW_BATCH = 128
OUT = os.environ.get("MG62_OUT", "mg62_multiaxis_fwd_nondividing.json")


# ── small helpers ────────────────────────────────────────────────────────────
def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, warmup=WARMUP, trials=TRIALS):
    """Milliseconds per call: warm-ups untimed and their results freed, then
    ``trials`` timed calls, each bracketed by a device sync and each result
    freed before the next allocation."""
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
    """Start a peak measurement and return the bytes already resident.

    ``reset_peak_memory_stats`` resets the high-water mark to what is LIVE now,
    so the counter read afterwards is resident + transient.  Several cells stay
    alive in this process, so the transient is the only figure comparable
    across them and across the dashboard, and it is resident subtracted off."""
    if not torch.cuda.is_available():
        return 0
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


def peak_mb(resident_bytes=0):
    """(peak MB including what was already resident, transient MB on its own)."""
    if not torch.cuda.is_available():
        return None, None
    peak = torch.cuda.max_memory_allocated()
    return peak / 2**20, (peak - resident_bytes) / 2**20


def fmt_mb(value):
    """Megabytes for a printed line; a CPU run has no CUDA counter to read."""
    return "n/a" if value is None else f"{value:.1f}"


def rel_max_err(a, b):
    """The project's scale-invariant gate: max|a - b| / max|b|."""
    a = a.detach().to(torch.float64)
    b = b.detach().to(torch.float64)
    denom = float(b.abs().max())
    return float((a - b).abs().max()) / max(denom, 1e-30)


# ── the cell under test ──────────────────────────────────────────────────────
class Cell:
    """One multiaxis cell, built exactly as the metrics harness builds it, with
    its inputs pre-placed on the device so a timed call measures the op."""

    def __init__(self, size):
        import mbirtorch
        self.size = tuple(int(x) for x in size)
        n_views, n_rows, n_channels = self.size
        angles = np.linspace(0, np.pi, n_views, endpoint=False)
        elevation = np.full(n_views, np.deg2rad(ELEVATION_DEG), dtype=np.float64)
        self.model = mbirtorch.MultiAxisParallelModel(
            self.size, np.column_stack([angles, elevation]))
        reference = mbirtorch.ParallelBeamModel(self.size, angles)
        self.model.set_params(recon_shape=reference.get_params('recon_shape'),
                              no_warning=True)
        self.model.set_params(verbose=0, no_warning=True)
        # The pin is mandatory and must precede any use: an unpinned model on a
        # multi-GPU node resolves lazily and can widen.
        self.model.configure_devices(num_devices=1)
        self.recon_shape = tuple(int(x) for x in
                                 self.model.get_params('recon_shape'))
        self.indices = mbirtorch.gen_full_indices(
            self.recon_shape, use_ror_mask=self.model.get_params('use_ror_mask'))
        self.num_pixels = int(len(self.indices))
        rng = np.random.default_rng(INPUT_SEED)
        cylinders = rng.standard_normal(
            (self.num_pixels, self.recon_shape[2]), dtype=np.float32)
        self.cylinders = self.model._shard_recon(cylinders)
        self.device = self.model.torch_device
        self.index_tensor = torch.as_tensor(self.indices, dtype=torch.int64,
                                            device=self.device)
        sync()

    @property
    def projectors(self):
        return self.model.projector_functions

    @property
    def forward_body(self):
        return self.projectors._fwd_body_per_dev[0]

    def body_args(self):
        return self.model._view_batch_args()

    def view_batch(self):
        """The view batch the driver itself would choose for the bound body."""
        args = self.body_args()
        return int(self.projectors._effective_view_batch(
            self.forward_body, self.num_pixels,
            int(self.cylinders.shape[-1]), args))

    def project(self):
        return self.model.sparse_forward_project(self.cylinders, self.indices)

    def label(self):
        return "x".join(str(x) for x in self.size)


# ── Part A: reproduce the dashboard numbers ──────────────────────────────────
def part_a(sizes):
    """One cell at a time, built and freed inside the loop, so the peak a row
    reports is not inflated by another cell's resident tensors.  The two focus
    cells are handed back for the later parts."""
    rows, kept = [], {}
    for size in sizes:
        cell = Cell(size)
        resident = reset_peak()
        times = timed(cell.project)
        peak, transient = peak_mb(resident)
        row = {"cell": cell.label(), "recon_shape": list(cell.recon_shape),
               "num_pixels": cell.num_pixels,
               "body": getattr(cell.forward_body, "__name__", str(cell.forward_body)),
               "view_batch": cell.view_batch(), "peak_mb": peak,
               "transient_mb": transient, **stats(times)}
        rows.append(row)
        print(f"  {row['cell']}: {row['min_ms']:.1f} ms min, "
              f"{row['median_ms']:.1f} ms median, peak {fmt_mb(peak)} MB "
              f"({fmt_mb(transient)} MB transient), body {row['body']}, "
              f"view batch {row['view_batch']}", flush=True)
        if tuple(size) in FOCUS_CELLS:
            kept[tuple(size)] = cell
        else:
            del cell
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows, kept


# ── Part B: where one projection's time sits ─────────────────────────────────
def part_b(cells):
    """The driver's own loop, run here so each body call can be timed.  This
    reproduces sparse_forward_project_view_range for the single-device case:
    same body, same arguments, same view slicing, same assembly."""
    rows = []
    for cell in cells:
        args = cell.body_args()
        body = cell.forward_body
        params = cell.projectors._view_params_per_dev[0]
        batch = cell.view_batch()
        num_views = cell.size[0]

        def one_call(first_view, count):
            return body(cell.cylinders, cell.index_tensor,
                        params[first_view:first_view + count],
                        slice_start=0, plan=None, **args)

        # Warm every distinct batch shape before timing, so no first-launch
        # compile lands in a timed call.
        for start in range(0, num_views, batch):
            out = one_call(start, min(batch, num_views - start))
            del out
        sync()

        calls = []
        sync()
        whole_start = time.perf_counter()
        assembled = None
        for start in range(0, num_views, batch):
            count = min(batch, num_views - start)
            sync()
            call_start = time.perf_counter()
            block = one_call(start, count)
            sync()
            call_ms = (time.perf_counter() - call_start) * 1e3
            if assembled is None:
                assembled = torch.empty((num_views,) + tuple(block.shape[1:]),
                                        dtype=block.dtype, device=block.device)
            copy_start = time.perf_counter()
            assembled[start:start + count] = block
            sync()
            copy_ms = (time.perf_counter() - copy_start) * 1e3
            calls.append({"views": count, "body_ms": call_ms,
                          "assemble_ms": copy_ms})
            del block
        sync()
        whole_ms = (time.perf_counter() - whole_start) * 1e3
        del assembled

        body_ms = sum(c["body_ms"] for c in calls)
        copy_ms = sum(c["assemble_ms"] for c in calls)
        row = {"cell": cell.label(), "view_batch": batch, "calls": calls,
               "n_calls": len(calls), "body_total_ms": body_ms,
               "assemble_total_ms": copy_ms, "loop_ms": whole_ms,
               "unattributed_ms": whole_ms - body_ms - copy_ms}
        rows.append(row)
        per_call = ", ".join(f"{c['views']}v {c['body_ms']:.1f}" for c in calls)
        print(f"  {row['cell']}: {row['n_calls']} body calls, "
              f"{body_ms:.1f} ms in bodies, {copy_ms:.1f} ms assembling, "
              f"{row['unattributed_ms']:.1f} ms elsewhere", flush=True)
        print(f"    per call (ms): {per_call}", flush=True)
    return rows


# ── Part C: what Triton reports about the two compiled kernels ───────────────
def compiled_variants():
    """Every compiled variant of the multiaxis forward kernel this process
    holds, with the fields Triton exposes.  The cache layout has moved between
    Triton versions, so several shapes are accepted."""
    from mbirtorch import triton_multiaxis
    kernel = triton_multiaxis._multiaxis_forward_kernel
    found = []

    def collect(store):
        if isinstance(store, dict):
            for key, value in store.items():
                if isinstance(value, dict):
                    collect(value)
                else:
                    found.append((key, value))

    for name in ("cache", "device_caches"):
        store = getattr(kernel, name, None)
        if store is not None:
            collect(store)
    rows = []
    for key, compiled in found:
        row = {"key": str(key)[:400]}
        for field in ("n_regs", "n_spills", "shared", "num_warps",
                      "num_stages", "name"):
            value = getattr(compiled, field, None)
            if value is None:
                value = (compiled.metadata.get(field)
                         if isinstance(getattr(compiled, "metadata", None), dict)
                         else getattr(getattr(compiled, "metadata", None), field, None))
            if value is not None:
                row[field] = value
        asm = getattr(compiled, "asm", None)
        if isinstance(asm, dict) and "ptx" in asm:
            row["ptx_lines"] = len(str(asm["ptx"]).splitlines())
        rows.append(row)
    return rows


def part_c():
    try:
        rows = compiled_variants()
    except Exception as error:                                   # noqa: BLE001
        print(f"  kernel introspection unavailable: {error!r}", flush=True)
        return [{"error": repr(error)}]
    for row in rows:
        print(f"  regs={row.get('n_regs')} spills={row.get('n_spills')} "
              f"shared={row.get('shared')} warps={row.get('num_warps')} "
              f"ptx_lines={row.get('ptx_lines')}  key={row['key'][:150]}",
              flush=True)
    return rows


# ── Part D: single-variable ablations at the body wrapper ────────────────────
def pad_band(values, width):
    """``values`` widened to ``width`` columns with zeros.  A column past
    num_slices is dropped from every tap by the kernel's own validity test, so
    widening changes the gather's row stride and nothing else."""
    if width == int(values.shape[1]):
        return values
    padded = torch.zeros((int(values.shape[0]), width), dtype=values.dtype,
                         device=values.device)
    padded[:, :int(values.shape[1])] = values
    return padded


def part_d(cells):
    from mbirtorch._utils import padded_kernel_width
    rows = []
    for cell in cells:
        args = cell.body_args()
        body = cell.forward_body
        params = cell.projectors._view_params_per_dev[0]
        batch = min(ABLATION_VIEW_BATCH, cell.size[0])
        view_params = params[:batch]
        values = cell.cylinders
        band = int(values.shape[1])
        pixels = cell.num_pixels
        pixels_16 = pixels - (pixels % 16)

        def call(values_arg, indices_arg, view_arg, **overrides):
            call_args = dict(args)
            call_args.update(overrides)
            return body(values_arg, indices_arg, view_arg, slice_start=0,
                        plan=None, **call_args)

        # The baseline every arm below is compared against: the cell's own
        # arguments at a fixed view batch.
        variants = [("baseline", dict())]
        # The two value-preserving band arms.  At the non-dividing cell the
        # band is padded UP to the next multiple of 16; at the dividing cell it
        # is stretched by one column, which is what the non-dividing cell has.
        # Both add zero columns past num_slices, so both must reproduce the
        # baseline's values.
        variants.append(("band=%d" % padded_kernel_width(band),
                         dict(_band=padded_kernel_width(band))))
        variants.append(("band=%d" % (band + 1), dict(_band=band + 1)))
        # The remaining arms change a bound and therefore change values; they
        # are timing probes only and their outputs are not compared.
        variants.append(("num_rows_r=%d" % padded_kernel_width(args['num_rows_r']),
                         dict(num_rows_r=padded_kernel_width(args['num_rows_r']))))
        variants.append(("num_rows_r=%d" % (int(args['num_rows_r']) - 1),
                         dict(num_rows_r=int(args['num_rows_r']) - 1)))
        variants.append(("num_channels=%d" % (int(args['num_channels']) - 1),
                         dict(num_channels=int(args['num_channels']) - 1)))
        variants.append(("num_slices=%d" % (int(args['num_slices']) - 1),
                         dict(num_slices=int(args['num_slices']) - 1)))
        variants.append(("num_pixels=%d" % pixels_16, dict(_pixels=pixels_16)))
        variants.append(("views=1", dict(_views=1)))

        # Some arms coincide with the baseline or with each other at a given
        # cell -- padding an already-divisible band, for one.  Keep the first
        # of each distinct argument set, so a cell's table has no repeats.
        seen, distinct = set(), []
        for name, spec in variants:
            key = (spec.get("_band", band), spec.get("_pixels", pixels),
                   spec.get("_views", batch),
                   tuple(sorted((k, v) for k, v in spec.items()
                                if not k.startswith("_"))))
            if key in seen:
                continue
            seen.add(key)
            distinct.append((name, spec))
        variants = distinct

        baseline_out = None
        cell_rows = []
        for name, spec in variants:
            spec = dict(spec)
            width = spec.pop("_band", band)
            count = spec.pop("_pixels", pixels)
            views = spec.pop("_views", batch)
            values_arg = pad_band(values[:count], width)
            indices_arg = cell.index_tensor[:count]
            view_arg = params[:views]

            def run(values_arg=values_arg, indices_arg=indices_arg,
                    view_arg=view_arg, spec=spec):
                return call(values_arg, indices_arg, view_arg, **spec)

            resident = reset_peak()
            times = timed(run)
            peak, transient = peak_mb(resident)
            row = {"cell": cell.label(), "variant": name, "views": views,
                   "band": width, "pixels": count, "peak_mb": peak,
                   "transient_mb": transient, **stats(times)}
            # The two band arms and the baseline share a value contract.
            if name == "baseline":
                baseline_out = run()
                row["rel_max_err_vs_baseline"] = 0.0
            elif name.startswith("band=") and count == pixels and views == batch:
                row["rel_max_err_vs_baseline"] = rel_max_err(run(), baseline_out)
            cell_rows.append(row)
            note = row.get("rel_max_err_vs_baseline")
            note = "" if note is None else f"  value diff {note:.2e}"
            print(f"  {row['cell']:>14} {name:<22} {row['median_ms']:8.2f} ms"
                  f"  transient {fmt_mb(row['transient_mb']):>8} MB{note}",
                  flush=True)
        del baseline_out
        base = next(r for r in cell_rows if r["variant"] == "baseline")
        for row in cell_rows:
            row["ratio_to_baseline"] = row["median_ms"] / base["median_ms"]
        rows.extend(cell_rows)
    return rows


# ── Part E: the candidate remedy, end to end ─────────────────────────────────
def banded_body(original):
    """``original`` with the values band padded up to a multiple of 16 before
    the call.  The padded columns sit past num_slices, so the kernel's validity
    test drops them from every tap and the values are unchanged."""
    from mbirtorch._utils import padded_kernel_width

    def padded_forward(values, pixel_indices, view_params_batch, *args, **kwargs):
        width = int(values.shape[1])
        launch = padded_kernel_width(width)
        if launch != width:
            values = pad_band(values, launch)
        return original(values, pixel_indices, view_params_batch, *args, **kwargs)

    padded_forward.__name__ = "band_padded_" + getattr(original, "__name__", "forward")
    # The driver reads the batching rule off the body, so it must ride along or
    # the remedy would silently change the view batch as well.
    cost = getattr(original, "_view_batch_cost", None)
    if cost is not None:
        padded_forward._view_batch_cost = cost
    padded_forward._mbirtorch_no_compile = True
    return padded_forward


def part_e(cells):
    rows = []
    for cell in cells:
        reference = cell.project()
        original = cell.projectors._fwd_body_per_dev[0]
        cell.projectors._fwd_body_per_dev[0] = banded_body(original)
        try:
            resident = reset_peak()
            times = timed(cell.project)
            peak, transient = peak_mb(resident)
            remedied = cell.project()
            row = {"cell": cell.label(), "peak_mb": peak,
                   "transient_mb": transient, "view_batch": cell.view_batch(),
                   "rel_max_err_vs_shipped": rel_max_err(remedied, reference),
                   **stats(times)}
            del remedied
        finally:
            cell.projectors._fwd_body_per_dev[0] = original
        del reference
        rows.append(row)
        print(f"  {row['cell']}: {row['min_ms']:.1f} ms min, "
              f"{row['median_ms']:.1f} ms median, peak {fmt_mb(peak)} MB "
              f"({fmt_mb(transient)} MB transient), "
              f"value diff {row['rel_max_err_vs_shipped']:.2e}", flush=True)
    return rows


# ── the run ──────────────────────────────────────────────────────────────────
def witnesses():
    import mbirtorch
    from mbirtorch import kernel_availability
    row = {"torch": torch.__version__,
           "mbirtorch_file": mbirtorch.__file__,
           "cuda_available": torch.cuda.is_available()}
    try:
        import triton
        row["triton"] = triton.__version__
    except Exception as error:                                   # noqa: BLE001
        row["triton"] = f"unavailable: {error}"
    if torch.cuda.is_available():
        row["device_name"] = torch.cuda.get_device_name(0)
        row["device_count"] = torch.cuda.device_count()
    row["probe_usable"] = list(kernel_availability.triton_available())
    return row


def main():
    report = {"witnesses": witnesses()}
    print("### witnesses")
    for key, value in report["witnesses"].items():
        print(f"  {key}: {value}")

    print("\n### Part A -- the dashboard numbers, through the library's funnel")
    report["part_a"], cells = part_a(CELLS)

    focus = [cells[tuple(size)] for size in FOCUS_CELLS]
    print("\n### Part B -- where one projection's time sits")
    report["part_b"] = part_b(focus)

    print("\n### Part C -- Triton's report on the compiled kernels")
    report["part_c"] = part_c()

    print("\n### Part D -- one integer changed at a time")
    report["part_d"] = part_d(focus)

    print("\n### Part E -- the band-padding remedy, end to end")
    report["part_e"] = part_e(focus)

    with open(OUT, "w") as sink:
        json.dump(report, sink, indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
