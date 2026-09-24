# Handoff: the device talk deck

Status: READY FOR THE TALK
Updated: 2026-09-24
Code: mbirtorch_plans main at 2a25b6e, pushed to origin
Next step: Greg gives the talk on 2026-09-25.  Everything after that is optional, and the choices are listed under "Open items".

## Summary

This folder holds the slides and demos for an informal talk to graduate
students on computing with several devices.  Greg Buzzard gives the talk on
2026-09-25.  He planned for 45 minutes.  Charlie and others will likely ask
many questions, so the talk may run to 2 hours.

The deck is complete.  It builds without errors, every slide fits its frame,
and all of its files were committed and pushed at 2a25b6e.  The deck has 25
numbered slides for the talk and 5 backup slides for questions.  The five
demos ran on Greg's M4 Max laptop on 2026-09-24, and the slides quote their
recorded results.

`README.md` describes the demos, the recorded results, and the source of
every figure.  This file covers the rest: the structure of the talk, the
decisions behind it, the open items, and how to edit and check the deck.

## The structure of the talk

The talk has one organizing idea.  A computer is a set of processors and
memories joined by links, and the links differ in speed by a factor of 50
or more.  Each section of the talk treats one part of that idea.

Greg proposed six topics: device types, threads, parallelism across devices
(vmap, SPMD, MPMD), memory, the levels of torch integration with the
hardware, and efficiency.  The deck covers all six, in a different order, and
adds a section on measuring honestly.

| slide | section | title | main sources |
|---|---|---|---|
| 1 | | What this talk covers | |
| 2 | The devices | Three kinds of device | Apple and NVIDIA specifications; `.claude/cluster_use.md`; `demo_machine.py` |
| 3 | | The links differ in speed by a factor of 50 | the same |
| 4 | Where the time goes | Four limits on speed | `demo_machine.py`; `demo_sizes.py` |
| 5 | | Arithmetic intensity decides between bandwidth and compute | `demo_machine.py`; H100 specification |
| 6 | | Compute-bound work scales with cores, and bandwidth-bound work does not | `demo_threads.py`; `.claude/lessons.md` Section 6 |
| 7 | Threads | One word, three meanings | general knowledge |
| 8 | | A GPU call returns before the GPU finishes | `demo_async.py` |
| 9 | | One Python thread per device | `mbirtorch/_sharding.py`; `mbirtorch/mace.py` |
| 10 | | torch's CPU threads: check how many you have | `mbirtorch/_host_threads.py`; `demo_threads.py` |
| 11 | One device, fully used | Each operation has a fixed cost, so give it enough data | `demo_sizes.py`; `mbirtorch/projectors.py` |
| 12 | | Four levels of control over the hardware | general knowledge |
| 13 | | Fusion reduces the trips through memory | `demo_fusion.py`; `mbirtorch/qggmrf.py` |
| 14 | | How mbirtorch uses the levels | `mbirtorch/kernel_availability.py`; `archive/torch_port/active/multigpu_findings.md` Sections 1.45 and 1.46 |
| 15 | Several devices | The first question: must the devices exchange data during a step? | |
| 16 | | Independent jobs: one worker per device, fed from a queue | `mbirtorch/mace.py`; `plans/mace4d/figures/fig6_devices.tex` |
| 17 | | Sharding: every device runs the same program on its part | `mbirtorch/_sharding.py` |
| 18 | | Sharding cuts the memory per GPU, and the time above a crossover size | `multigpu_findings.md`; `.claude/lessons.md` Section 6 |
| 19 | | When all the GPUs together are not enough, split the problem | `plans/parallel_4k/plan.md` |
| 20 | Memory | Where GPU memory goes | `surveys/leap_comparison/findings/ornl_reproduction.md`; `plans/device_memory_tiers/plan.md` |
| 21 | | Copies: move data once, in large pieces, from pinned memory | `surveys/leap_comparison/findings/host_gather.md` |
| 22 | | Memory and time trade against each other | |
| 23 | Measuring honestly | Rules for timing and memory measurements | `.claude/lessons.md` Section 5; `demos/devtime.py` |
| 24 | | Five measurements that misled us | `.claude/lessons.md` Section 5; `.claude/cluster_use.md` |
| 25 | Summary | Summary | |

The backup slides follow a frame titled "Backup slides", and their numbers
restart at 1.  They are, in order: the same ideas in JAX and torch, getting
GPUs on gautschi, very large arrays and the 2^31 boundary, precision on each
device, and the measured numbers for the M4 Max.  The section title pages
carry no slide numbers.

## Decisions and their reasons

Each decision below was discussed with Greg or follows his style guides:

- Efficiency comes second, not last.  The four limits on speed are overhead,
  memory bandwidth, compute, and capacity.  Every later section uses them as
  its vocabulary.
- vmap appears with batching on one device (slide 11).  It turns a loop into
  one batched operation, and it does not spread work across devices.
- SPMD and MPMD appear through one question: must the devices exchange data
  during a step?  The answer selects one of three patterns: independent jobs
  fed from a queue, sharding, or splitting a problem into pieces.
- The section on measuring honestly is new.  Its stories come from
  `.claude/lessons.md` Section 5.
- The examples come from the group's own code and records wherever possible.
  Every slide that gives a number has a `\source` line that names its record.
- The slide text follows `.claude/writing_style.md` and
  `.claude/writing_style_charlie.md`, because Charlie will be in the
  audience.  The text uses complete sentences, no dashes as punctuation, no
  chains of clauses joined by semicolons, and no metaphors.
- The figure PDFs in `images/` are committed, as in the update deck.  Two
  charts come from `../mbirtorch_update_2026-09/images/` through
  `\graphicspath`: `memory_levels.pdf` and `gather_time.pdf`.

## Measurement caveats

The demo numbers have four limits, which questions may raise:

- The thread demo varied between runs by up to 15 percent.  In the first run,
  sin on 12 threads was 11.1 times faster than on one thread.  In the
  recorded run it was 9.4 times faster, and slide 6 says "about 9 times".
- `demo_threads.py` times the matrix product in its own sweep, after add and
  sin.  When the product ran between them, the add and sin timings were
  noisier.  The likely cause is the threads of Apple's Accelerate library,
  which runs the CPU matrix product outside torch's thread pool.
- The H100 numbers are NVIDIA specification values, and the slides say so.
  The M4 Max bandwidth of 546 GB/s is Apple's specification.  The other M4
  Max numbers are measured.
- The first compiled call on MPS took 0.12 s, but it ran after the CPU
  compile in the same process.  It is therefore not a clean compile time.

## Open items

None of these items is needed before the talk.  Each one is Greg's decision:

1. Some statements come from general knowledge, not from the group's
   records.  Greg may want to check them before the talk: the
   JAX and torch table (backup slide 1), the TF32 behavior of
   `torch.set_float32_matmul_precision('high')` (backup slide 4), GPU threads
   running in groups of 32 (slide 7), and Triton running on NVIDIA and AMD
   GPUs (slide 12).
2. The slides are fairly dense, because they use complete sentences.  Greg
   may trim them for speaking.
3. The demos run unchanged on a gautschi H100 node, but `make_figures.py`
   assumes the M4 Max results.  Three places would need changes before
   charting another machine.  The device label `'mps'` appears in
   `operation_size()` and `roofline()`, and the efficiency-core shading at
   threads 12.5 to 16 appears in `thread_scaling()`.  After those changes,
   set `HOST` to the name in the new CSV files.
4. Every run of `make_figures.py` changes the four committed PDFs, even when
   the charts do not change.  matplotlib and pdflatex write a creation date
   into each PDF.  `savefig(..., metadata={'CreationDate': None})`, and
   `SOURCE_DATE_EPOCH` for pdflatex, would make reruns identical.
5. LaTeX still reports small overfull boxes.  The largest is 26 pt on slide
   6, and the title page has 14 pt.  The pages were rendered to images and
   inspected, and no text is cut off.
6. After the talk, questions that exposed gaps could become backup slides.

## How to build and check

The deck builds on Greg's Mac with MacTeX, and the demos use the `mbirtorch`
conda environment (torch 2.14.0 with MPS).  Run these commands from this
folder:

```bash
/opt/miniconda3/envs/mbirtorch/bin/python demos/demo_async.py   # each demo writes demos/results/<demo>_apple_m4_max.csv
/opt/miniconda3/envs/mbirtorch/bin/python make_figures.py       # writes images/*.pdf, with pdflatex for the MACE figure
latexmk devices_talk_deck.tex                                    # the PDF builds here, other files in tex_output/
grep 'Overfull \\vbox' tex_output/devices_talk_deck.log          # frames whose content runs past the bottom
gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -r80 -sOutputFile=<scratch dir>/p%02d.png devices_talk_deck.pdf
```

Three cautions apply to these commands:

- Rerunning a demo overwrites its CSV file.  The slide text quotes numbers
  from those files, so the text must be updated when they change.
- poppler's `pdftoppm` is not installed on the Mac.  Ghostscript, at
  `/usr/local/bin/gs`, renders the pages for a visual check.
- A cloud session can edit the slides and, with texlive installed, build the
  deck.  It cannot rerun the demos, because the cloud machines have no GPU.

## LaTeX and shell details found while building the deck

Each of these details cost time once.  They are recorded here so that the
next session avoids them:

- A frame body must not start with a brace group, such as `{\small ...}`.
  Beamer reads that group as the frame's subtitle, and the metropolis theme
  does not show subtitles, so the content disappears.  Start the body with
  `\vspace{0pt}`.  The header of the `.tex` file records this rule.
- A `\section` after `\appendix` fails with "Arithmetic overflow" when
  `appendixnumberbeamer` is loaded.  The theme's section page divides by the
  frame count.  The deck uses a standout frame for "Backup slides" instead.
- A frame that contains `lstlisting` must be `[fragile]`, and the code lines
  must start in column 0.
- In `\texttt` with the T1 FiraMono font, `--` prints as an en dash.  Write
  `-{}-` for a command-line option such as `-{}-mem`.
- Math in a frame title needs `\texorpdfstring`, so that the PDF bookmark
  stays readable.
- A metropolis frame holds less text than one might estimate from its size.  About 100
  words of `\small` text fit in a half-width column beside a figure.  A
  listing line in a half-width column should stay under about 45 characters.
- Charts drawn at 4.6 by 3.3 inches stay legible at half the slide width.  At
  6.0 by 3.9 inches their text was too small to read.
- In zsh, `echo` interprets backslash escapes, so `echo '\begin'` writes a
  backspace character.  Write LaTeX with the file tools or with Python.
- In zsh, a glob that matches nothing stops the whole command.  A word that
  starts with `=` is expanded as a command path, so `echo ===` fails.

## Pointers

The following files hold Greg's preferences and the related work:

- `.claude/claude_prompt.md`: how Greg wants a session to collaborate.
- `.claude/writing_style.md` and `.claude/writing_style_charlie.md`: the
  writing rules the slides follow.
- `../mbirtorch_update_2026-09/`: the deck whose look this one follows, and
  the source of two of its charts.
