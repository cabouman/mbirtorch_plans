# Remaining high-value time and memory investigations

Ranked by expected payoff against cost, in three groups.


**Directly on the new forward path (cheap, immediate):**

1. **Overlap the column transfers with the compute.** The driver currently fetches each batch's columns, waits, projects, repeats. Last night's rows show the gap: at four devices with batch 4096, GPUs were busy 12.4 s inside a 16.3 s window — about a quarter of the forward wall is transfer stall, shrinking to ~7% at batch 32768. Prefetching the next batch while the current one projects is already permitted by the per-device execution structure and was written up as its own increment; it costs one extra batch of resident memory (the memory model term is already parameterized for it). Likely the best remaining ratio of effort to time saved.
2. **Cheapen the per-batch accumulation.** Each batch's output is added into the full sinogram shard; at small batches that add is pure overhead. Preallocate and accumulate less often. Small, pairs naturally with the overlap work.
3. **Measure the two geometries that were left out.** Translation and multi-axis still use the slice-band path and share the cost structure that made cone slow there. The record deliberately says they switch only on their own measurement — one cheap two-arm job per geometry could hand them the same 1.2–1.6x.
4. **Re-run the pixel-batch sweep at production size.** 8192 stayed the default because the sweep that favored 16384–32768 ran at the 1024 class; the 2K-and-larger sweep is what should actually set it.

**Time, beyond this path:**

5. **The unexplained kernel-width behavior.** A projection kernel launch costs the same whether it covers a half-width or full-width block of values — that flatness is why narrow calls waste, and nobody knows the mechanism (occupancy? launch overhead? memory latency?). Understanding it is the entry point for making the kernels themselves faster, including the single-device case every configuration pays. One cheap discriminating arm is already specified: parallel at one device with the band forced to half width.
6. **Re-evaluate the sorted-channel reduction.** mbirjax sorts each call's detector-channel writes in-kernel and reduces by segment; we declined it at the old speeds. The forward just got 1.4–2x faster, which moves the threshold that decision rested on — the recorded plan is to probe the light per-call form first.
7. **The cone back-projection rise at two devices** (30.3 s where the four-device case is far better) is still unowned and unexplained.

**Memory (all at the 2K+ production scale, where it binds):**

8. **The back-projection reduce term.** Each GPU's partial results plus the running total sit on the owner device at about 1.5 full cylinders, and that cost is flat in device count — adding GPUs does not shrink it, which makes it the structural per-GPU memory ceiling at 2K. Restructuring that reduce (tree-shaped, or in-place) is the one identified change that would let memory actually scale down with device count.
9. **Validate the memory model at 2K.** Every calibration so far ran at the 1024 class; the first composed 2K runs both check the ledger where it matters and anchor the production-scale tiling work (which also inherits the cache questions you approved: phase-blocked accumulation and the L2 residency probes).

Housekeeping already queued, not investigations: the device-count threshold re-measurement after your commit, and a one-arm check of the hand-written CUDA kernels at single-pixel calls.

If I had to pick two: the transfer overlap (1) for near-term time, and the reduce-term restructuring (8) for whether 2K fits comfortably per GPU.

# Sorted-stream and 2D tiling

They're both in the list, but under other names — let me place them explicitly.

**Sorted-stream is item 6, plus a deeper form inside the kernel work.** What I called "re-evaluate the sorted-channel reduction" is exactly the sorted-stream idea: sort each call's detector-channel writes so the scatter becomes ordered, collision-free segments instead of random adds. It has two stages with different costs. The light form — sort and segment-reduce within each call, then write once — is the queued cheap probe: it was declined earlier against the old forward's numbers, and the forward getting 1.4–2x faster moves that threshold, so the recorded plan is to re-read the gate on the new numbers before anything bigger. The full form — the sorted stream accumulating on-chip rather than through global memory — is one of the four cache directions you approved, and it belongs inside the kernel campaign because it rewrites the kernel interior. Note the coupling to item 5: the width mystery's discriminating arm effectively decides sorted-stream's prospects. If a launch costs the same regardless of width because the kernel is launch-overhead-bound, sorting inside the call fixes nothing; if it's memory-collision-bound, sorted-stream is the remedy pointed directly at the cause. That one cheap arm informs both.

**2D tiling is the substance of item 9's follow-on — the production-scale program.** The column path we just made default is one-dimensional tiling: it blocks the pixel axis and deliberately keeps the full slice height, because cone pays full height per call no matter what. 2D tiling is the next claim — block both axes of a call's working set so it stays resident in L2 while it's being accumulated, which is your recollection of where the large effect was. Its case has to be made at 2K: the 1K measurements mostly refuted L2-residency effects at that size (one ~10% anomaly aside, the working sets either fit trivially or miss regardless), so the honest sequence is the item-9 baseline runs at 2K first — they establish where time and memory actually go at production size — and tiling is the main lever if those runs show cache-boundedness or a memory squeeze. Phase-blocked accumulation, the first cache direction you approved, is the entry-level version of the same idea.

**Why neither reorders the list:** both change the kernel interior or its blocking, beneath the same column-batched data flow that just landed. The driver, the batch knob, the memory-model terms, and the default all survive them — which was the answer to your earlier worry about tuning too tightly. The near-term items (transfer overlap, accumulation, the two unmeasured geometries) are cheap and independent of both, so they go first; sorted-stream's light-form re-gate and the width arm are the cheap probes that then decide how much of the expensive kernel work — including full sorted-stream and 2D tiling — is actually warranted, with tiling's verdict waiting on 2K data rather than 1K extrapolation.