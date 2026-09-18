# mbirtorch update deck, September 2026

`mbirtorch_update_deck.tex` describes the features added to mbirtorch since
release v0.0.2 (2026-08-21), and then the high-value features of LEAP and
TIGRE that mbirtorch still lacks.  It follows the look of the pcdrecon decks
(`research_notes/decks/` in that repository).

Build with `latexmk mbirtorch_update_deck.tex` from this directory.  The
`latexmkrc` here puts the PDF beside the source and every other build file
in `tex_output/`.  The deck reads its figures from `images/`.

## Figures

`make_figures.py` writes the four charts as PDF into `images/` from numbers
in the records it names, copies two images from their sources, and draws the
fusion image from the cluster's slice arrays.  The repository ignores PNG
files, so the two copied images live only in the working tree; the fusion
image is committed, because its source is on the cluster:

| image | source |
|---|---|
| `memory_layers.pdf` | `plans/device_memory_tiers/experiments/dm1_record.md`, the verification table |
| `memory_levels.pdf` | `surveys/leap_comparison/findings/ornl_reproduction.md`, Section 3.2, the four-device row |
| `ornl_gpu_memory.pdf` | `surveys/leap_comparison/findings/ornl_reproduction.md`, Section 3.1 |
| `gather_time.pdf` | `surveys/leap_comparison/findings/host_gather.md`, the summary table |
| `geometry_viewer_cone.png` | `docs/source/figs/geometry_viewer_cone.png` in the mbirtorch repository |
| `quality_nrmse_vs_time_512.png` | `surveys/leap_comparison/experiments/results/` |
| `ornl_fusion_zoom.png` | written by `make_figures.py <path>` from the slice arrays `/scratch/gautschi/buzzard/leap_ornl/out/msf/ornl_sigma002_slices.npz` on gautschi (keys `standard`, `postproc`, `fusion`), from the run `surveys/leap_comparison/experiments/ornl/msf_ornl.md` records; a 240-voxel window at the top edge of the part |

To redraw the fusion image, copy the `.npz` file from the cluster to the Mac
and pass its path as the script's one argument.  Without the argument the
script writes or copies the other six images and leaves the committed fusion
image as it is.
