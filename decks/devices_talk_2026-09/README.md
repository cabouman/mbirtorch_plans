# Computing on several devices, talk of September 25, 2026

`devices_talk_deck.tex` is an informal talk for graduate students on using
CPUs, GPUs, and Apple's MPS from torch: the devices and the links between
them, the four limits on speed, threads, compiling and hand-written kernels,
patterns for several devices, memory, and honest measurement.  Its examples
come from mbirtorch and from five demos in `demos/`.  It follows the look of
the update deck in `../mbirtorch_update_2026-09`.

Build with `latexmk devices_talk_deck.tex` from this directory.  The
`latexmkrc` here puts the PDF beside the source and every other build file in
`tex_output/`.  Slides 1 to 25 are the talk; the slides after "Backup slides"
are for questions.

## Demos

Each demo runs on a Mac with MPS or on a machine with a CUDA GPU, with no
change.  Run it from any directory with a Python that has torch; it prints a
table and writes `demos/results/<demo>_<cpu model>.csv`.

| script | what it shows |
|---|---|
| `demo_async.py` | a GPU call returns before the GPU finishes; `synchronize()` and `.item()` wait |
| `demo_machine.py` | the matrix product rate, the memory bandwidth, the cost of one small operation, and the host-GPU copy rates, per device |
| `demo_sizes.py` | the time of one addition against the array size: a fixed cost for small arrays, the bandwidth for large ones |
| `demo_threads.py` | CPU speedup against torch's thread count, for a bandwidth-bound add and a compute-bound sin; the CPU matrix product for comparison |
| `demo_fusion.py` | eager torch against `torch.compile` on the qGGMRF surrogate weight of `mbirtorch/qggmrf.py` |

`devtime.py` holds the helpers they share: the device choice, the
synchronization, and the timing function the deck shows.

## Results on the M4 Max

Measured on September 24, 2026, on an Apple M4 Max (12 performance and 4
efficiency cores, a 40-core GPU, 128 GB) with torch 2.14.0 in the `mbirtorch`
conda environment.  The CSV files in `demos/results/` hold every number.

- An 8192 by 8192 matrix product on MPS returned after 0.11 ms and finished
  after 81.9 ms, a rate of 13.4 TFLOP/s.
- The MPS memory bandwidth for `c = a + b` was 432 GB/s, and the CPU's was
  316 GB/s.  These rates count two reads and one write per value.  A copy
  from host to MPS ran at 89 GB/s, and back at 44 GB/s.  The copy rates
  count each array's bytes once.
- One small operation cost 3.3 microseconds on MPS when issued without a
  wait.  With a wait after each one, it cost about 140 microseconds.
- The CPU matrix product ran at about 3.0 TFLOP/s with every thread count
  from 1 to 16, because torch sends it to Apple's Accelerate library.
- On 12 threads, sin ran 9.4 times faster than on one thread, and add 2.5
  times faster.  An earlier run of the same sweep gave 11.1 and 2.7; the
  deck uses the recorded run.  With 14 and 16 threads the efficiency cores
  slowed sin down.
- The qGGMRF weight on 2^26 values took 15.8 ms eager and 1.42 ms compiled on
  MPS, and 96.8 ms and 38.2 ms on the CPU.  The first compiled call on the
  CPU took 9.9 s.

## Figures

`make_figures.py` draws the three charts from the demo results and builds
the MACE figure.  The deck also reads two charts from the update deck's
`images/` folder, through its graphics path.

| image | source |
|---|---|
| `images/roofline.pdf` | `demos/results/machine_apple_m4_max.csv`, and the NVIDIA H100 SXM specification (67 TFLOP/s float32, 3.35 TB/s) |
| `images/thread_scaling.pdf` | `demos/results/threads_apple_m4_max.csv` |
| `images/operation_size.pdf` | `demos/results/sizes_apple_m4_max.csv` and `machine_apple_m4_max.csv` |
| `images/mace_devices.pdf` | built from `plans/mace4d/figures/fig6_devices.tex` |
| `memory_levels.pdf` | `../mbirtorch_update_2026-09/images/`, from `surveys/leap_comparison/findings/ornl_reproduction.md` |
| `gather_time.pdf` | `../mbirtorch_update_2026-09/images/`, from `surveys/leap_comparison/findings/host_gather.md` |

To chart another machine, run the demos there, copy the CSV files into
`demos/results/`, and set `HOST` in `make_figures.py` to the name in their
file names.
