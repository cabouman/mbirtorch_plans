# Gathering a slice-sharded volume to the host

## Summary

The gather of a reconstruction from several CUDA devices to the host ran at
about a fifth of the rate the hardware allows.  A reconstruction is sharded
on its last axis, the slices, so each device's shard lands in a strided
block of the one host array: every (row, column) pair receives a run of that
shard's slices.  `Shards.gather` copied the shards into those blocks one
device after another.  Torch carries out such a copy by staging the whole
shard in pageable host memory and then writing it into place with one
thread.  That host write is the slow part.  The transfer itself is fast.
A shard crosses into pinned host memory at 54 GB/s per device, and at 94
to 139 GB/s with two to four devices at once.

`Shards.gather` now moves every CUDA shard at once, each in slabs of 16 MiB
through two pinned staging slots per host thread, with several host threads
writing each shard's block.  The thread count is the CPUs the process may
run on divided by the shard count, at most sixteen per shard.  Torch's own
intra-op threads are divided out as well, because the write is torch's copy
and already runs on them.  Nothing else changes: the result is the same
C-contiguous host array, and shards on other devices take the old path.
The pinned memory held is two slots per thread, and torch keeps those slots
cached after the first gather.

The table gives warm figures on the ORNL volume (1360 x 1360 x 1296
float32, 8.9 GiB) on gautschi H100 nodes.  The after column is the changed
library at its shipped settings, taken from the `library_cap16` rows of jobs
16428659 and 16428660.  The tree those jobs ran carried the thread cap at
eight, and the shipped constant is sixteen.

| devices | before | after | pinned transfer alone |
|---|---|---|---|
| 1 (28 CPUs) | 2.3 s | 0.25 s | 0.18 s |
| 2 (28 CPUs) | 4.8 s | 0.26 s | 0.10 s |
| 4 (56 CPUs) | 5.7 s | 0.24 s | 0.07 s |

The gather now runs at 36 to 40 GB/s, which is 9 to 24 times faster than
before and within a factor of 1.4 to 3.4 of the pinned transfer alone.
The remaining time is the host write itself.  With 14 threads per shard
the write alone takes 0.21 to 0.23 s on these nodes.

## 1. Where the time went

The FDK split in `surveys/leap_comparison/findings/ornl_reproduction.md` (Section 3.4) put the volume gather
at 7.1 to 8.0 s on four H100s, against 9.2 s for the back projection that
produced the shards.  The harness `gather_bench.py` times the gather and its
parts on the same volume, each variant twice with a new host array each
time.  It checks every result element for element against a plain
concatenation of the shards.

Three reference figures bound the cost.  A contiguous copy of each shard
into pinned host memory measures the transfer alone.  A copy of touched host
copies of the shards into their strided blocks of a new host array measures
the host write alone, page faults included.  A copy into a host array that
was zeroed before the timer started measures the host write without the
page faults.  The table gives the four-device figures from the run of the
unchanged library (job 16427948):

| variant | seconds | GB/s |
|---|---|---|
| `Shards.gather` as it was | 5.75 | 1.7 |
| the same per-shard copy, one thread per device | 1.70 | 5.6 |
| pinned slabs, one host thread per shard | 1.13 | 8.5 |
| host write alone, one thread per shard | 1.09 | 8.8 |
| host write with the array zeroed first | 0.79 | 12 |
| pinned transfer alone, four devices at once | 0.07 | 139 |
| pageable copy of each shard (`tensor.cpu()`), four at once | 0.62 | 15 |

The gather was bound by one host thread per device writing a strided block,
not by the transfer.  Running the four copies at once made it 3.4 times
faster, and staging through pinned memory on top of that made it 5 times
faster.  That variant ran at the same speed as the host write alone.  The
page faults of the new host array were about a third of that write.

Host threads per shard are the remaining lever.  The strided write ran at
about 5 GB/s per thread and sped up nearly in proportion to the threads.
The table gives job 16428335, a two-device allocation with 28 CPUs, and job
16428336, four devices with 56 CPUs:

| threads per shard | 1 device, 28 CPUs | 2 devices, 28 CPUs | 4 devices, 56 CPUs |
|---|---|---|---|
| 1 | 1.79 s | 1.39 s | 1.07 s |
| 2 | 0.94 s | 0.73 s | 0.57 s |
| 4 | 0.51 s | 0.43 s | 0.36 s |
| 7 | 0.36 s | 0.33 s | 0.29 s |
| 14 | 0.27 s | 0.25 s | 0.24 s |
| host write alone, 14 threads | 0.22 s | 0.22 s | 0.21 s |
| pinned transfer alone | 0.18 s | 0.10 s | 0.07 s |

The gain flattens between 7 and 14 threads per shard, where the write
approaches the host write alone and the transfer alone.  With 14 threads
per shard the gather runs at 35 to 40 GB/s, against the 10 GB/s the
earlier record measured for a pageable copy of the sinogram to the
devices.  The cap of sixteen threads per shard lets every CPU of these
allocations take part.  On the two-device node (job 16428659) a cap of
eight gave 0.30 s in place of 0.25 s, and a cap of four gave 0.43 s.

The slot size matters little above 16 MiB.  At 14 threads per shard on two
devices, 4 MiB slots took 0.33 s, 16 MiB slots 0.25 s, and 64 MiB slots
0.25 s.  Large slots have a cost of their own.  The first gather in a
process pins its slots, and 28 threads with 64 MiB slots pinned 3.5 GiB in
about a second before the first copy.  Slots of 8 MiB were up to 11 percent
slower than 16 MiB in the run of the changed library (job 16428660).  So
the library uses 16 MiB.

Torch's own intra-op threads do the same work when it has them.  With
torch set to 14 intra-op threads and one host thread per shard, the two-
device gather took 0.27 s, and adding four host threads per shard on top
took 0.22 s.  So the two forms of parallelism do not conflict.  These jobs
started with `OMP_NUM_THREADS=1` in the environment, which is why torch
reported one thread in every run here.

## 2. What was rejected

Transposing each shard on its device to slice-major order makes the host
block contiguous.  That variant wrote the volume in 0.54 s on four devices
with one thread per shard, twice as fast as the strided write.  It
was not adopted because the result would be in (slices, rows, columns)
order, and transposing it back on the host would cost a second full-size
array and another pass over the volume.

A numpy assignment in place of torch's copy for the host write ran at the
same speed with two or more threads and 20 percent slower with one, so the
torch copy stays.

## 3. The change

`_sharding.py` gains five helpers and two settings.  `_gather_threads_per_shard`
gives the threads per shard.  `_split_grid` cuts a shard's block into one
row range per thread, or column ranges when the block has a single row.
`_slab_pieces` cuts a thread's range into pieces that fit a slot.
`_copy_through_slots` moves one thread's range through two slots, issuing
the copy of the next slab before writing the current one into place and
waiting on an event per slab.  `_fill_from_shards_in_slabs` runs every
thread's range of every shard in one thread pool.  `GATHER_SLOT_BYTES` is
the slot size and `GATHER_THREADS_PER_SHARD` the cap on threads per shard.
`Shards.gather` takes this path when every shard is on a CUDA device and the
old path otherwise, after the same shape checks as before.

The tests in `test_sharding.py` exercise the slab path on CPU tensors over
these cases: slot sizes from one element up, one to five threads per shard,
both axes and a middle axis, uneven splits, a shard of zero length, and
three element sizes.  A CUDA-only test runs the same cases on the real
devices with small slots and three threads per shard.  The harness's own
correctness pass ran `Shards.gather` on the H100s against the host arrays
it was given: both axes, uneven and zero-length shards, three element
sizes, and slots down to 64 bytes.  It matched in 162 cases on four devices
and 108 on two (jobs 16428660 and 16428659), and the CUDA test passed on
both nodes.

## 4. Files

- Harness: `surveys/leap_comparison/experiments/gather/gather_bench.py`,
  with its batch files and the logs and results of jobs 16427947, 16427948
  (library as it was), 16428335, 16428336 (thread sweep), and 16428659, 16428660
  (library as changed) in `results/`.  Logs, results and batch files are
  ignored by git, as for the rest of this directory.  The tables above
  carry every number they hold.
- Library change: `mbirtorch/_sharding.py`, `tests/test_sharding.py`, and
  one sentence in `docs/source/dev_sharding_overview.rst`.

## 4. Addendum: the gather when torch has all its threads

The measurements above were taken with torch on one host thread, which is
what a job on gautschi had when it sourced the cluster preamble.  The
preamble now leaves torch its threads, and the pool rule of section 3
divides the cores by torch's thread count, so under the new preamble the
gather runs with one host thread per shard and torch's parallel copy does
the strided write.  Three conditions were timed, `Shards.gather` on the
tree at 6eef8bf, the 9.6 GB volume of the ORNL scan, two repetitions each
(jobs 16564867, 16564866, and 16581010; the first repetition on one device
pays the page faults of the new host array):

| condition | 1 device | 2 devices | 4 devices |
|---|---|---|---|
| torch one thread, 14 gather threads per shard | 0.27 to 0.42 s | 0.25 s | 0.27 to 0.30 s |
| torch 56 threads, one gather thread per shard (the rule as shipped) | 0.64 to 0.72 s | 0.37 s | 0.33 s |
| torch 56 threads, the gather setting torch to one thread for its duration | 0.25 to 0.40 s | 0.24 s | 0.24 to 0.26 s |

Torch's own copy parallelizes the strided write less well than the explicit
threads, so the shipped rule gives up 0.08 s on four devices, 0.13 s on
two, and 0.3 to 0.4 s on one, once per reconstruction, in proportion to the
volume.  The third row was a change that made the gather run under a
temporarily single-threaded torch and restored the count afterward.  It
was measured and not adopted: it flips a process-wide setting from inside
the library for a gain that is small against any reconstruction, and the
gather with threads free is still 17 times faster than before the rework.
The rule of section 3 stands.  Result files:
`experiments/gather/results/gather_bench_{16564866,16564867,16581010}.json`.
