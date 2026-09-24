# Rust GUIs in mbirtorch, plan of record

Status: PROPOSED
Updated: 2026-09-24
Code: nothing built. This plan describes mbirtorch prerelease 02f3498 and cabouman/mbirjax pull request 225 as of 2026-09-24.
Next step: Greg and Charlie take the three decisions at the end of this plan, and then Greg consults Jean Bilheux about moving the GUI.

Date: 2026-09-24.  Decision recorded on 2026-09-24 (Greg): mbirjax is no longer active.  The
Rust GUI code therefore lives in a `gui/` folder of the mbirtorch repository and is released
with mbirtorch.

Paths.  mbirtorch paths are given from the root of the mbirtorch repository.  Paths inside
the pull request are given from the root of its crate, `dehydration_hydration_ui/`, in the
mbirjax repository.  All other paths are in this repository.

## Executive summary

mbirtorch gains an optional set of GUIs written in Rust.  The first GUI is the dehydration
GUI that Jean Bilheux wrote for the VENUS beamline at ORNL.  This GUI denoises a stack of
neutron images with `mbirtorch.hsnt.hyper_denoise`.  That function dehydrates the data by
projecting each pixel spectrum onto a low-dimensional subspace, and then it rehydrates the
data.  The GUI is in cabouman/mbirjax pull request 225, which is not merged.

The design has four parts.

- **Distribution.**  The GUIs ship as a second PyPI package, `mbirtorch-gui`, built from the
  `gui/` folder.  PyPI serves this package as compiled wheels for Linux and macOS, so users
  need no Rust compiler.  Users install it with `pip install "mbirtorch[gui]"`.  mbirtorch
  itself stays pure Python.
- **One program.**  The package installs one program, `mbirtorch-gui`, and each GUI is a
  subcommand of that program.  A new GUI is a new Rust module and a new subcommand.  It needs
  no new package and no new release step.
- **A worker in the library.**  The program never reimplements the science in the library.
  Instead, the program runs the library in a separate Python process, called the worker.  The
  program and the worker exchange arrays through `.npy` files.
- **The viewers stay.**  The matplotlib slice viewer and geometry viewer remain the default.
  A Rust slice viewer is decided later.  The decision waits until the dehydration GUI shows
  whether Rust GUIs work on the group's remote displays.

The work has four increments, and each increment is reviewed before the next one starts.

1. The dehydration GUI moves into `gui/`, together with the worker and their tests.
2. The release workflow publishes the GUI package to TestPyPI and PyPI.
3. The dehydration GUI is measured on the group's two remote display routes.
4. Greg and Charlie decide whether to build a Rust slice viewer.

## Status by increment

| Increment | Delivers | Status |
|---|---|---|
| 1 | The dehydration GUI in `gui/`, the worker `mbirtorch/gui_worker.py`, and their tests | Not started |
| 2 | The GUI package on TestPyPI and PyPI, built by the release workflow | Not started |
| 3 | A findings page on the dehydration GUI over the two remote display routes | Not started |
| 4 | The decision on a Rust slice viewer, recorded in this plan | Not started |

## Rule for the code

Nothing from this plan goes into the code.  Comments, docstrings, and the README of `gui/`
describe what the code does.  They do not name this plan, its increments, or the history of
the pull request.  This rule applies to the Rust code and to the Python code.

## Purpose

This plan has three goals.

1. VENUS users get the dehydration GUI through a standard install.  mbirtorch now carries
   `hsnt`, so the GUI moves to mbirtorch.
2. The group gets a way to build GUIs where matplotlib is weak.  The existing viewers stay in
   Python unless a later decision moves them.
3. Users who never open a GUI see no change.  mbirtorch stays one pure Python wheel, and
   cluster installs stay the same.

## What exists today

### The library

mbirtorch is pure Python.  Its release workflow, `.github/workflows/release.yml`, builds one
wheel that installs on every platform.  The workflow uploads pre-releases to TestPyPI, and it
uploads releases to PyPI after approval.  Both uploads use trusted publishing, which needs no
stored token.

`pixi.toml` declares two platforms, linux-64 and osx-arm64.  It also sets the minimum system
versions that the torch wheels need: glibc 2.28 on Linux and macOS 14.

`mbirtorch/hsnt.py` holds `hyper_denoise`, `dehydrate`, `rehydrate`, and the private
estimator `_estimate_subspace_dimension`.  The module imports numpy, h5py, scipy, and
scikit-learn, but it does not import torch.  Importing the module still runs
`mbirtorch/__init__.py`, which imports torch.  On Greg's Mac, `import mbirtorch.hsnt` took 13 s
on its first run and 1.6 s on later runs.  The modules that `hsnt.py` imports took 0.9 s by
themselves.  These times were measured on 2026-09-24 in the conda environment `mbirtorch`,
with torch 2.13.0.

Each viewer separates a pure numpy model from a matplotlib view.  In the slice viewer, the
model `VolumeStack` and the view `SliceViewer` are both in `mbirtorch/viewers/slice_figure.py`.
In the geometry viewer, the model `GeometryScene` is in `mbirtorch/viewers/geometry_scene.py`,
and the view `GeometryFigure` is in `mbirtorch/viewers/geometry_figure.py`.  The geometry
viewer also has a browser version, whose sources are in `plans/geometry_viewer/experiments/web/`.

`archive/viewer/slice_viewer_eval.md` compared GUI toolkits for the slice viewer and chose
matplotlib.  Remote display decided the ranking.  The evaluation named PySide6 with pyqtgraph
as an optional frontend to add later on the same model.  It ranked OpenGL viewers last,
because OpenGL over forwarded X is usually broken or slow on clusters.

`.claude/cluster_use.md` gives two routes for showing a window from a gautschi compute node.
Route A forwards X to the Mac's X server over `ssh -Y`.  Route B forwards X to a ThinLinc
desktop on a login node.  Both routes reach the compute node with `srun --x11`.

No Rust toolchain is installed on Greg's Mac.

### The pull request

cabouman/mbirjax pull request 225, titled "224 dehydration hydration UI", was opened by Jean
Bilheux on 2026-08-15.  It comes from the branch `224_dehydration_hydration_ui` of
JeanBilheux/mbirjax, and it is not merged.  It adds one Rust crate,
`dehydration_hydration_ui/`, in 23 files.  The command `gh pr diff 225 -R cabouman/mbirjax`
retrieves it.

The GUI covers the whole correction workflow.  It loads a folder of TIFF images and runs the
correction on a background thread, with a cancel button.  It shows the corrected and raw
stacks side by side with shared contrast, or it shows their difference.  It plots the mean
spectrum of a region that the user draws.  It exports the corrected stack as float32 TIFF
files with a provenance file, `correction_config.json`.  A `--run` option runs the same steps
without a window.

The GUI is built with egui 0.35 and its application framework, eframe.  The crate's lock file
contains no GTK packages.  The lock file does contain `x11-dl` and `xkbcommon-dl`.  These
crates load the X11 libraries when the program runs, instead of linking them.

The GUI calls Python through a bridge.  The bridge is a Rust module, `src/py_bridge.rs`, and a
Python script, `python/run_hyper_denoise.py`.  The module writes the data as a `.npy` file in a
temporary folder and runs the script.  The script loads `mbirjax/hsnt.py` by file path from the
repository checkout, which avoids importing jax.  The script reports progress in lines of
standard output that start with `##`.  Cancel kills the Python process.  The module finds
Python through the variable `MBIRJAX_PYTHON`, and then through `python3` or `python` on the
PATH.  It finds the script through the variable `MBIRJAX_HSNT_BRIDGE`, a path compiled into the
program, or the program's own folder.

The crate also carries a Rust port of the algorithm.  The files `src/hsnt.rs`, `src/nmf.rs`,
and `src/linalg.rs` hold 1,044 lines.  The program does not use them.  They serve unit tests
and a comparison example, `examples/cross_check.rs`.

Three properties of the bridge need to change.

1. The bridge depends on the repository checkout.  The script path is compiled into the
   program, and the script loads `hsnt.py` by file path.  An installed program has neither the
   path nor the checkout.
2. The Auto button estimates the number of materials with its own copy of the preprocessing in
   `dehydrate`.  For transmission data, the copy skips the first denoising pass that
   `dehydrate` applies before it takes the log.  Both mbirjax and mbirtorch apply this pass.
   The Auto estimate can therefore differ from the estimate that `dehydrate` makes itself.
3. The bridge does not pass `random_state` to `hyper_denoise`.  The docstring of
   `hyper_denoise` states that the factors then vary from run to run.  A run therefore cannot be
   repeated exactly from its provenance file.

## The design

### Distribution

The GUI package is a second distribution, built from the `gui/` folder of the mbirtorch
repository.  The folder holds the Rust crate and a `pyproject.toml` whose build backend is
maturin.  maturin is the standard tool that builds Rust code into Python wheels.  With the
setting `bindings = "bin"`, maturin puts the compiled program into the wheel.  pip installs the
program into the environment's `bin` folder, which also holds that environment's `python`.
ruff and uv are distributed this way.  Rerun's PyPI wheels also carry a viewer built with egui.

The release builds two wheels, one for Linux x86_64 and one for macOS arm64.  The Linux wheel
follows manylinux_2_28, the PyPI rule set for Linux wheels that run on glibc 2.28 or newer.
The torch wheels set the same glibc minimum.  These two wheels cover the platforms in
`pixi.toml`.  Intel Macs need no wheel.  The last torch release with Intel Mac wheels was 2.2,
and the library requires torch 2.13 or newer.

The manylinux rules let a wheel link only a short list of system libraries.  egui loads the
X11 and OpenGL libraries when the program runs, so it can meet the rules.  A program that links
GTK or Qt would have to carry that toolkit inside the wheel.  maturin checks the rules when it
builds the Linux wheel, so Increment 2 confirms the result.

The GUI package publishes no source distribution.  On a platform without a wheel, pip then
reports that no matching distribution exists.  pip does not try to compile the Rust code.

The versions of the two packages move together.  The GUI package has the same version as the
library, and it requires exactly that version of the library.  The library's `gui` extra names
the GUI package without a version.  pip then selects the GUI package whose version matches the
library.

The release workflow gains one job.  The job builds the two wheels with the maturin GitHub
action, `PyO3/maturin-action`.  It builds the Linux wheel on `ubuntu-latest` in a
manylinux_2_28 container, and it builds the macOS wheel on `macos-latest`, which has an arm64
processor.  The existing publish jobs upload these wheels with the library's files.  The
existing tag check also compares the tag with the version of the GUI package.  Greg registers
`mbirtorch-gui` for trusted publishing on TestPyPI and PyPI, with the same workflow and
environments as the library.

On macOS, pip does not mark the files it downloads as quarantined.  The program therefore runs
without Apple's notarization, as ruff and uv do.

### One program with one subcommand per GUI

The GUI package installs one program, and each GUI is a subcommand of that program.  The
dehydration GUI starts with `mbirtorch-gui dehydration`.  The `gui/` folder has this layout.

```text
gui/
  pyproject.toml      maturin settings and package metadata
  Cargo.toml          the crate and its dependencies
  README.md
  src/main.rs         reads the subcommand and starts that GUI
  src/common/         code that more than one GUI uses, starting with the worker client
  src/dehydration/    the dehydration GUI
  tests/              tests that need no display
```

One program keeps the cost of a new GUI small.  egui and the libraries it uses make up most of
the program's size, and one program compiles them once.  Each platform needs one wheel, and all
GUIs share one version.  A new GUI adds a folder under `src/` and a subcommand.  It needs no new
PyPI package, workflow job, or trusted publisher.

Code moves into `src/common/` when a second GUI needs it.  The first candidate is the image
stack panel of the dehydration GUI.  This panel shows stacks side by side with shared contrast,
shows their difference, and lets the user select a region.  A slice viewer would need the same
panel.

### Rust for display, Python for science

The program never reimplements the science in the library.  Science here means the algorithms,
the geometry conventions, and the preprocessing that goes with them.  Display logic belongs in
the program.  Examples of display logic are slicing, contrast, difference images, and region
statistics.  With one implementation of each algorithm, the program and the library cannot
compute different results for the same algorithm.

The rule has two consequences for the pull request.

- The Rust port and `examples/cross_check.rs` are removed.  Tests of the worker replace their
  tests.
- The Auto estimate uses the library's preprocessing.  Increment 1 moves the preprocessing in
  `dehydrate` into a private function of `mbirtorch/hsnt.py`.  `dehydrate` and the estimate
  task of the worker both call that function.  For transmission data, the function includes
  the first denoising pass, so the Auto estimate takes longer than it does now.

### The worker

The worker is a module of the library, `mbirtorch/gui_worker.py`, started as
`python -m mbirtorch.gui_worker`.  Placing the worker in the library has three effects.

- The worker imports the installed library, so it needs no file paths.
- The worker is released with the functions it calls, so it can call private functions safely.
- pytest can test the worker without Rust and without a display.

The program starts one worker when a GUI opens, and the worker runs until the GUI closes.  The
import time measured above is then spent once, while the user opens data.  Cancel kills the
worker, and the program starts a new one.

The program looks for Python in this order.

1. The interpreter named by the variable `MBIRTORCH_PYTHON`, which serves development.
2. The `python` in the program's own folder, which is where pip puts both of them.
3. `python3` on the PATH.

The worker and the program exchange JSON messages, one message per line.  The program writes
requests to the standard input of the worker.  The worker writes its messages to its original
standard output, and it redirects everything else to standard error.  Output printed by the
library therefore cannot corrupt the messages.

The first message from the worker states the protocol version, the library version, and the
names of its tasks.  Because the two packages share one version, a mismatch is rare.  A program
built from source can still meet a different installed library.  The program therefore checks
the protocol version and reports a mismatch.

A request names a task, its parameters, and the paths of its input and output `.npy` files.  The
worker answers with progress messages and then one result or error message.  The first tasks are
`hsnt.denoise` and `hsnt.estimate`.  A new GUI that needs the library adds tasks to the worker.
The messages could look like the following lines.  The specification of Increment 1 fixes their
fields.

```text
worker:  {"type": "ready", "protocol": 1, "mbirtorch": "0.2.0", "tasks": ["hsnt.denoise", "hsnt.estimate"]}
program: {"id": 1, "task": "hsnt.denoise", "params": {"num_materials": 2, "random_state": 0},
          "inputs": {"data": "input.npy"}, "outputs": {"denoised": "output.npy"}}
worker:  {"type": "progress", "id": 1, "fraction": 0.15, "stage": "denoising"}
worker:  {"type": "result", "id": 1, "values": {"subspace_dimension": 4}}
```

Arrays cross as `.npy` files in a temporary folder that the program creates and deletes.  Rust
reads `.npy` files through the crate `ndarray-npy`, which needs no C library.  The pull request
already uses this crate.  The Rust HDF5 crates bind the HDF5 C library, and the wheel would then
have to carry that library.

Every request that uses randomness carries a `random_state`.  The program sends a fixed seed by
default, and it records the seed in the provenance file.  A run can then be repeated exactly.

### Two ways to start a GUI

A GUI can start in two ways.  In the first way, a user runs `mbirtorch-gui dehydration` in a
desktop session, and the program starts the worker.  The dehydration GUI starts only in this
way.  In the second way, a Python function writes arrays to `.npy` files and starts the
program.  A Rust slice viewer would start in this way, for example through
`slice_viewer(vol, backend="native")`.  Its window would belong to a separate process, so it
would never block the calling script.

This plan builds only the first way.  Every GUI accepts its input files and options on the
command line, and the second way needs nothing more from the program.

### The viewers

The matplotlib viewers stay the default, and they do not change.  They are pure Python, they
need no compiled dependency, and they save figures.  They also work over forwarded X, which is
how the group sees windows from cluster nodes.

The slice viewer would benefit most from a Rust version.  A Rust slice viewer would likely
redraw slices faster than matplotlib.  No measurement of that difference exists yet.  Each window of a Rust viewer would
be a separate process that never blocks a script.  Dialogs built with egui would replace the Tk
dialogs of the matplotlib viewer.  The Rust viewer would reimplement the display logic of
`VolumeStack`, which the rule for science allows.  The image stack panel of the dehydration GUI
would be its starting point.

Remote display is the main risk.  egui draws through a GPU interface, either OpenGL or wgpu.
wgpu is a Rust graphics library that runs on Vulkan, Metal, or OpenGL.  The toolkit evaluation ranked OpenGL viewers last for remote display.  Increment 3 therefore
measures the dehydration GUI on both display routes before any work on a viewer starts.

The alternative for the slice viewer is still the runner-up of the evaluation, PySide6 with
pyqtgraph.  This alternative keeps the viewer in Python.  Its cost is a Qt dependency, which the
evaluation put at more than 100 MB.

A Rust geometry viewer is not planned.  The science of the geometry viewer is in
`GeometryScene`, which stays in Python, so a Rust version would only redraw its output.  The
geometry viewer also has a browser version already.

## Increments

Each increment is carried out from a written specification, reviewed, and staged by explicit
file name.

1. **The dehydration GUI in mbirtorch.**  Move the crate into `gui/`, rename the program
   `mbirtorch-gui`, and give the dehydration GUI the subcommand `dehydration`.  Replace the
   bridge with a worker client, and write `mbirtorch/gui_worker.py` with the two tasks.  Move the
   preprocessing in `dehydrate` into the shared private function.  Remove the Rust port.  Greg's
   Mac has no Rust toolchain, so the increment first installs one, from rustup or from
   conda-forge through a pixi feature.  Gate: `cargo test` and the new pytest tests pass on the
   Mac.  An end-to-end test runs `mbirtorch-gui dehydration --run` on a synthetic TIFF stack made
   with `mbirtorch.hsnt.generate_hyper_data`.  The test compares the exported stack with
   `hyper_denoise` called directly with the same seed.  The test states its tolerance and the
   reason for that tolerance.  Jean confirms that the GUI still works on VENUS data.
2. **The GUI package on PyPI.**  Add `gui/pyproject.toml`, the `gui` extra, the release job, and
   the extended tag check.  Greg registers the trusted publisher on TestPyPI and PyPI.  Add a
   short section on the GUI to `docs/source/install.rst`.  Gate: a pre-release tag publishes both
   packages to TestPyPI.  In a new environment on the Mac and on gautschi,
   `pip install "mbirtorch[gui]"` from TestPyPI installs the program.  The command
   `mbirtorch-gui dehydration --run` then completes on the synthetic stack.  The same holds in a
   pixi environment on the Mac.
3. **Remote display.**  Run the installed dehydration GUI on a gautschi compute node over both
   routes.  Record whether the window opens, which renderer egui uses, and the frame time while a
   slider moves over a 2048 by 2048 image.  Record the same numbers on the Mac as the local
   reference.  Gate: a findings page in `findings/`.
4. **The viewer decision.**  Greg and Charlie use the evidence of Increments 1 to 3 to choose
   among three options for the slice viewer.  The options are matplotlib only, a Rust viewer
   started from Python, and a pyqtgraph viewer.  A Rust viewer or a pyqtgraph viewer gets its
   own plan.  Gate: the decision is recorded in this plan.

Increments 1 and 2 deliver the dehydration GUI to users.  Increments 3 and 4 decide whether
the group builds more Rust GUIs.

## Decisions for Greg and Charlie

1. **Who moves the code.**  Jean can open a new pull request against mbirtorch that follows
   Increment 1.  Alternatively, a Claude session can move the code on a branch for Jean to
   review.  In both cases, a Claude session writes the worker and the change to
   `mbirtorch/hsnt.py`, because they are library code.  The git history keeps Jean as the author
   of the moved Rust code.  Recommendation: ask Jean which of the two ways to use.  Jean knows
   the VENUS data and can test the GUI under ThinLinc at ORNL.
2. **Names.**  The PyPI package and the program are both named `mbirtorch-gui`.  This name was
   free on PyPI on 2026-09-24.  The first subcommand is `dehydration`, from the title of the GUI.
   Recommendation: use these names.
3. **Platforms.**  The release builds wheels for Linux x86_64 and macOS arm64, the platforms in
   `pixi.toml`.  Windows and Linux arm64 would each add one wheel to every release.
   Recommendation: build the two wheels now, and add others when a user asks for them.

## Risks

Three risks remain after the design.

- **Remote display.**  The GUI may be unusable over forwarded X on gautschi.  Rust GUIs would
  then suit only displays where they run directly, such as a laptop.  Increment 3 measures this
  risk.
- **A second language.**  Fewer people in the group can change Rust code than Python code.  The
  rule for science limits the Rust code to display, and Claude sessions can make routine
  changes.
- **Release time.**  Each release builds two more wheels, so the release workflow takes longer.

## Out of scope

This plan does not cover a Rust geometry viewer, a conda-forge package, Windows wheels, or a
browser build of the GUIs.  It builds nothing for starting a GUI from Python.  Memory-mapped
viewing of large volumes belongs to `plans/memory_map_viewer/`.
