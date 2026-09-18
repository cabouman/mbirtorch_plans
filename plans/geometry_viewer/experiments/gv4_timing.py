"""Time a slider step and a trajectory toggle on an 1800-view helical scan.

Why this measurement.  The geometry viewer plan sets one gate for Increment 4:
a slider step must redraw an 1800-view model in under 100 ms.  A scan that
size is an ordinary helical acquisition, and a viewer whose slider lags by half
a second is not usable for the checking-a-scan-before-reconstruction case the
plan is built around.

What is timed.  The script builds a ``ConeBeamModel`` of 1800 views from
parameters alone, with no projection and no data, builds a ``GeometryFigure``
under the Agg backend, and then times three things: the scene's own
``trajectory`` over all views, twenty ``set_view`` steps, and one trajectory
toggle.  A ``set_view`` step includes the redraw, because the redraw is what
the user waits for.  Under Agg that redraw is the partial-redraw path the
viewer uses on an interactive backend as well: the cached background is
restored, the artists that moved are drawn, and the region is blitted.

Two figures are timed, so that the fast path can be compared with the plain
one.  The first allows the partial redraw.  The second is built with
``blit=False``, which makes every step repaint the whole figure, and is the
reference the optimization is measured against.

There are no command-line arguments.  The parameters are the constants below.

Run:
    cd plans/experiments/geometry_viewer
    MPLBACKEND=Agg PYTHONPATH=<mbirtorch clone> python gv4_timing.py

The run takes about a minute and prints one table.  ``gv4_timing.md`` holds the
numbers of a run and the machine they were measured on.
"""

import os
import sys
import time

import numpy as np
import matplotlib

# The figures are never shown, so no window is needed.  This has to happen
# before pyplot resolves a backend, which the viewer defers to its first
# figure.
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mbirtorch  # noqa: E402
from geometry_scene import GeometryScene  # noqa: E402
from geometry_viewer import GeometryFigure  # noqa: E402

# ── run parameters ───────────────────────────────────────────────────────────

#: The number of views.  This is the plan's gate size.
NUM_VIEWS = 1800

#: The detector shape, (rows, channels).  A real helical scan's detector is
#: larger, and the detector's size costs the drawing nothing, because the
#: drawing draws its outline and not its pixels.
DETECTOR_SHAPE = (40, 64)

#: The reconstruction shape, (rows, columns, slices).
RECON_SHAPE = (24, 24, 16)

#: The scan: the two distances in ALU, the number of turns the source makes,
#: and the total helical travel along z in ALU.
SOURCE_DETECTOR_DIST = 400.0
SOURCE_ISO_DIST = 200.0
NUM_TURNS = 2.0
HELICAL_TRAVEL = 30.0

#: How many slider steps are timed, and how far apart they are.  The steps are
#: spread over the scan so that each one changes every drawn position.
NUM_STEPS = 20
STEP_STRIDE = 89

#: The gate, in milliseconds per slider step.
GATE_MS = 100.0

#: The figure size in inches, the same one the render scripts use.
FIGSIZE = (15.0, 9.0)


def build_model():
    """The 1800-view helical cone-beam model, from parameters only.

    Compilation is off, so no projector is compiled and the run stays on the
    CPU.  No sinogram and no reconstruction are ever made: the viewer reads
    parameters through ``get_params`` and nothing else.
    """
    angles = np.linspace(0.0, NUM_TURNS * 2.0 * np.pi, NUM_VIEWS,
                         endpoint=False)
    z_shifts = np.linspace(0.0, HELICAL_TRAVEL, NUM_VIEWS)
    shape = (NUM_VIEWS,) + tuple(DETECTOR_SHAPE)
    model = mbirtorch.ConeBeamModel(
        shape, angles, source_detector_dist=SOURCE_DETECTOR_DIST,
        source_iso_dist=SOURCE_ISO_DIST, helical_z_shifts=z_shifts,
        compile_mode='off')
    model.set_params(no_warning=True, recon_shape=RECON_SHAPE)
    model.verify_valid_params()
    return model


def time_calls(function, count):
    """Call ``function(index)`` ``count`` times and return the times in ms."""
    times = []
    for index in range(count):
        start = time.perf_counter()
        function(index)
        times.append((time.perf_counter() - start) * 1e3)
    return np.asarray(times)


def time_figure(scene, blit):
    """Time the slider steps and one trajectory toggle of one figure.

    Args:
        scene (GeometryScene): the geometry to draw.
        blit (bool): whether the figure may use the partial-redraw fast path.

    Returns:
        dict: the timings in milliseconds.
    """
    import matplotlib.pyplot as plt

    start = time.perf_counter()
    figure = GeometryFigure(scene, figsize=FIGSIZE, blit=blit)
    build_ms = (time.perf_counter() - start) * 1e3

    # One step before the measurement, so that the background is cached and
    # the first measured step is like every later one.
    figure.set_view(1)
    steps = time_calls(
        lambda index: figure.set_view((index + 1) * STEP_STRIDE % NUM_VIEWS),
        NUM_STEPS)

    start = time.perf_counter()
    figure.set_show_trajectory(True)
    toggle_on_ms = (time.perf_counter() - start) * 1e3
    with_path = time_calls(
        lambda index: figure.set_view((index + 1) * STEP_STRIDE % NUM_VIEWS),
        NUM_STEPS)
    start = time.perf_counter()
    figure.set_show_trajectory(False)
    toggle_off_ms = (time.perf_counter() - start) * 1e3

    plt.close(figure.figure)
    return dict(build_ms=build_ms, steps=steps, with_path=with_path,
                toggle_on_ms=toggle_on_ms, toggle_off_ms=toggle_off_ms)


def main():
    """Build the model, time both figures, and print the table."""
    start = time.perf_counter()
    model = build_model()
    model_ms = (time.perf_counter() - start) * 1e3

    start = time.perf_counter()
    scene = GeometryScene.from_model(model)
    scene_ms = (time.perf_counter() - start) * 1e3

    start = time.perf_counter()
    quantities = scene.derived_quantities()
    quantities_ms = (time.perf_counter() - start) * 1e3

    trajectory_times = time_calls(lambda _index: scene.trajectory(), 5)

    print(f'{NUM_VIEWS} views, detector {DETECTOR_SHAPE}, '
          f'recon {RECON_SHAPE}, backend {matplotlib.get_backend()}')
    print(f'helical travel {quantities["helical_travel_alu"]:.1f} ALU, '
          f'magnification {quantities["magnification"]:.2f}')
    print('')
    print(f'{"step":40s} {"mean ms":>9s} {"max ms":>9s}')
    print(f'{"build the model":40s} {model_ms:9.1f} {"":>9s}')
    print(f'{"build the scene":40s} {scene_ms:9.1f} {"":>9s}')
    print(f'{"scene.derived_quantities()":40s} {quantities_ms:9.1f} '
          f'{"":>9s}')
    print(f'{"scene.trajectory()":40s} {trajectory_times.mean():9.1f} '
          f'{trajectory_times.max():9.1f}')

    for blit in (True, False):
        name = 'partial redraw' if blit else 'full repaint'
        result = time_figure(scene, blit)
        print('')
        print(f'{"build the figure, " + name:40s} '
              f'{result["build_ms"]:9.1f} {"":>9s}')
        for label, times in (('set_view step, ' + name, result['steps']),
                             ('set_view step with path, ' + name,
                              result['with_path'])):
            print(f'{label:40s} {times.mean():9.1f} {times.max():9.1f}')
            verdict = 'passes' if times.max() < GATE_MS else 'FAILS'
            print(f'{"    against the " + str(int(GATE_MS)) + " ms gate":40s} '
                  f'{verdict:>9s}')
        print(f'{"trajectory toggle on, " + name:40s} '
              f'{result["toggle_on_ms"]:9.1f} {"":>9s}')
        print(f'{"trajectory toggle off, " + name:40s} '
              f'{result["toggle_off_ms"]:9.1f} {"":>9s}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
