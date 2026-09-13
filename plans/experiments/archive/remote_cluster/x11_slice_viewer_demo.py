"""X11 smoke test: run a small mbirtorch recon ON A GPU NODE and display it with
slice_viewer on the user's screen (ssh -Y to a Mac X server, or ThinLinc via srun --x11).

Prints a diagnostic block FIRST so that if no window appears we can tell which link
failed (no GPU / DISPLAY not forwarded / non-interactive backend) instead of guessing.

Small by design (64 views, 96x90 detector): every slice_viewer redraw ships pixels over
the network, so this is about proving the path works, not about a realistic workload.

Ported from the mbirjax original on 2026-09-03; the mbirjax version was verified end to
end on 2026-07-25, this one has not yet been run on a display.
"""
import os
import sys

print("=" * 68, flush=True)
print("NODE       :", os.uname().nodename, flush=True)
print("DISPLAY    :", os.environ.get("DISPLAY", "<UNSET -- X11 not forwarded>"), flush=True)
print("XAUTHORITY :", os.environ.get("XAUTHORITY", "<unset>"), flush=True)

import torch
print("torch      :", torch.__version__, " cuda available:", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    print("\nFATAL: no CUDA device -- this must run on a GPU node, not a login node.", flush=True)
    sys.exit(2)
print("GPU        :", torch.cuda.get_device_name(0), flush=True)

# mbirtorch's viewer imports pyplot lazily and takes whatever backend matplotlib picks:
# TkAgg with a DISPLAY, Agg without one (then show() warns and draws nothing).  Resolve the
# backend now, before the recon, so a missing display fails here and not after the work.
import matplotlib
import matplotlib.pyplot  # noqa: F401  (forces backend resolution)
backend = matplotlib.get_backend().lower()
print("mpl backend:", backend, flush=True)
if backend in ("agg", "pdf", "ps", "svg", "template", "cairo"):
    print("\nFATAL: backend is non-interactive -- no window can appear.", flush=True)
    print("       Either DISPLAY is not set on this node, or TkAgg failed to load.", flush=True)
    sys.exit(3)

# Prove the GUI toolkit can actually open a connection before spending time on a recon.
try:
    import tkinter
    _root = tkinter.Tk()
    _root.withdraw()
    print("tkinter opened a display connection OK", flush=True)
    _root.destroy()
except Exception as e:
    print("\nFATAL: tkinter could not open the display:", type(e).__name__, e, flush=True)
    sys.exit(4)
print("=" * 68, flush=True)

import mbirtorch
print("mbirtorch  :", mbirtorch.__file__, flush=True)

# ── the actual work: small cone-beam demo data + a short recon, on the GPU ──────
print("\ngenerating demo data (small) ...", flush=True)
phantom, sinogram, params = mbirtorch.generate_demo_data(
    object_type='shepp-logan', model_type='cone',
    num_views=64, num_det_rows=96, num_det_channels=90)
angles = params['angles']
print("  sinogram", sinogram.shape, " phantom", phantom.shape, flush=True)

ct_model = mbirtorch.ConeBeamModel(sinogram.shape, angles,
                                   params['source_detector_dist'], params['source_iso_dist'])
ct_model.set_params(sharpness=1.0)
weights = mbirtorch.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

print("reconstructing (max_iterations=10) ...", flush=True)
recon, recon_dict = ct_model.recon(sinogram, weights=weights, max_iterations=10)
print("  recon", recon.shape, flush=True)
mbirtorch.get_memory_stats()

print("\nopening slice_viewer -- the window should appear on your screen.", flush=True)
print("(this blocks until you close the window)", flush=True)
mbirtorch.slice_viewer(
    phantom, recon, slice_label=['Phantom', 'MBIR recon'],
    title='mbirtorch on {} -- displayed via X11'.format(os.uname().nodename))
print("slice_viewer closed cleanly.", flush=True)
