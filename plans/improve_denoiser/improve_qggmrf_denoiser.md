# Starting prompt: fix the scaling of QGGMRFDenoiser.denoise and evaluate a faster solver

Read these two files before you start, and follow them throughout:

- `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_plans/.claude/claude_prompt.md`
- `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_plans/.claude/writing_style.md`

If the work needs the cluster, also read `.claude/cluster_use.md` in the same
folder before the first job.

## Summary

A test in the pcdrecon repository found two problems in
`QGGMRFDenoiser.denoise`.  The test used mbirtorch 0.0.2.  Both problems are
still present in the mbirtorch checkout at commit 63207f7, which is version
0.1.1.

1. **Scaling.**  The update of each pixel uses 1 as the second derivative of
   the data term, instead of \( 1/\sigma_y^2 \).  As a result, `denoise`
   barely changes an image whose noise standard deviation \( \sigma_y \) is
   much larger than 1, such as an image in Hounsfield units (HU).  A one-line
   change in each of three functions fixes the problem.
2. **Convergence.**  With the scaling fixed, the default of 15 iterations
   still stops far from the MAP estimate.  On a test slice at mbirtorch's
   automatic strength, the fixed solver needed 420 iterations and 7.5 seconds
   to come within 0.01 HU of the MAP estimate.  L-BFGS came as close in 47
   iterations and 0.5 seconds.

Your job has two parts.  First, confirm the scaling problem with a test, and
fix it.  Second, evaluate whether a faster solver or a different stopping rule
should replace the current ones.  As claude_prompt.md asks, summarize your
understanding and propose a plan before you edit code.  Report to Greg before
you change a default or the algorithm.

## The cost

`denoise` approximates the MAP estimate, which is the image \( x \) that
minimizes

$$
J(x) = \frac{\lVert y - x \rVert^2}{2 \sigma_y^2} + \sum_{\lbrace i, j \rbrace} b_{ij} \rho(x_i - x_j) ,
$$

$$
\rho(\Delta) = \frac{|\Delta|^p}{p \sigma_x^p} \frac{u^{q - p}}{1 + u^{q - p}} , \quad u = \frac{|\Delta|}{T \sigma_x} .
$$

Here \( y \) is the noisy image, and \( b_{ij} \) are the neighbor weights.
The sum counts each pair of neighbors once.  With mbirtorch's default
weights, each pair of nearest neighbors has \( b_{ij} = 1/6 \)
**[checked in 2D]**.  For \( 1 \le q \le p \le 2 \), \( J \) is convex, so
the MAP estimate is unique.

The strength of the prior is set by the ratio \( r = \sigma_x / \sigma_y \).
A smaller \( r \) smooths more.  mbirtorch's automatic regularization chose
\( r = 0.10 \) on the synthetic slice used below, and \( r = 0.14 \) on a
slice of a real CT image.

## The scaling problem

`vcd_subset_denoiser` updates a subset of pixels at once.  Each pixel \( i \)
in the subset moves along the direction

$$
\delta_i = - \frac{g_i + g^{prior}_i}{h_i + h^{prior}_i} .
$$

In this direction, \( g_i = (x_i - y_i) / \sigma_y^2 \) is the first
derivative of the data term, and \( h_i = 1/\sigma_y^2 \) is its second
derivative.  \( g^{prior}_i \) and \( h^{prior}_i \) are the same derivatives
of the prior's surrogate.  The code computes \( g_i \) correctly, as
`forward_grad = -fm_constant * cur_error_image`, with `fm_constant` equal to
\( 1/\sigma_y^2 \).  However, the code sets `forward_hess = 1` where
\( h_i \) belongs.  A line search then multiplies the direction by a step
factor \( \alpha \).  The line search uses the correct data term, but it
clamps \( \alpha \) to at most 1.5 (`max_alpha`).

The effect of the wrong \( h_i \) depends on \( \sigma_y \).  To see this,
write the prior's second derivative as \( h^{prior}_i = \kappa_i / \sigma_y^2 \).
Then the code's direction is the correct direction multiplied by

$$
\frac{1 + \kappa_i}{\sigma_y^2 + \kappa_i} .
$$

For a 2D image with the default weights, \( \kappa_i \) is at most
\( 2/(3 r^2) \), which is 67 at \( r = 0.1 \) **[derived]**.  The factor
behaves differently in two regimes:

- **\( \sigma_y \) much larger than 1.**  For an image in HU with
  \( \sigma_y = 182 \), the factor is at most 1/490 at \( r = 0.1 \).  Even
  with \( \alpha = 1.5 \), each step is at least 300 times shorter than the
  correct step.  In a test, the released code clamped \( \alpha \) at 1.5 in
  each of 60 iterations **[checked]**.
- **\( \sigma_y \) much smaller than 1.**  For an image in attenuation units,
  the factor is between 1 and \( (1 + \kappa_i)/\kappa_i \), and the line
  search can shrink \( \alpha \) as needed.  A test found no effect in this
  regime.  The test divided the synthetic slice by \( 10^5 \), so that
  \( \sigma_y = 0.0018 \).  At \( r = 0.1 \), the distances of the released
  and fixed code to the MAP estimate agreed within 2% at each of six
  iteration counts from 5 to 400.  At \( r = 0.8 \), both came within
  0.001 HU of the MAP estimate in 15 iterations **[checked]**.  So MACE and
  plug-and-play loops on images in attenuation units are not affected
  **[inferred]**.

At 63207f7, the problem is in three functions:

| Function | Second derivative of the data term | Clamp of \( \alpha \) |
|---|---|---|
| `vcd_subset_denoiser` | line 104, `forward_hess = 1` | lines 123 and 124 |
| `vcd_subset_denoiser_batched` | line 159, `forward_hess = 1` | lines 177 and 178 |
| `_denoise_sharded` | line 848, `(1.0 + hess)` | line 866 |

The fix replaces 1 by `fm_constant` in each function:

```python
forward_hess = fm_constant                                 # vcd_subset_denoiser and vcd_subset_denoiser_batched
delta = -((forward_grad + grad) / (fm_constant + hess))    # _denoise_sharded
```

With this fix, \( \alpha \) was 1.000 in each of 60 iterations at
\( r = 0.1 \), so the clamp no longer limits the step **[checked]**.

mbirjax 0.7.3 has the same line in both of its denoiser updates, at lines 457
and 522 of `denoising.py` **[checked in the wheel from PyPI]**.  So mbirtorch
likely inherited the problem from mbirjax **[inferred]**.

## Evidence

The tests used mbirtorch 0.0.2 and a synthetic slice of 500 by 500 pixels in
HU, with water at 1000 and with iodine, muscle and bone inserts.  The slice's
noise has a standard deviation of 182 HU per pixel.  Its noise power rises
with frequency, as in an FDK image.  A reference MAP estimate came from L-BFGS
in float64, with an error below 0.001 HU rms **[derived]**.  Each distance
below is the rms distance to this reference over all pixels.

The tests support three statements:

1. **The released code barely moves an image in HU.**  With its defaults,
   `denoise` stopped after 1 iteration, because the image changed by less than
   its stopping threshold.  The noise fell from 181.6 to 181.5 HU.  Even 1000
   iterations moved the image only from 155.8 to 137.6 HU from the MAP
   estimate.
2. **The fix makes the result independent of the image's unit.**  The fixed
   code on the image in HU followed the same path as the released code on the
   image divided by 1000.  At each of eight iteration counts from 1 to 400,
   the two distances to the MAP estimate agreed within 1.2%.  After 1000
   iterations, both distances were 0.003 HU.
3. **The cost above is the cost that `denoise` minimizes.**  The pcdrecon test
   compared its L-BFGS solution with the released `denoise`, run 3000
   iterations on the image divided by 1000.  At five settings, on the
   synthetic slice and on a real slice, the two agreed within 0.02 HU at every
   pixel.

## Convergence and the stopping rule

`denoise` stops after `max_iterations=15`, or earlier when
\( 100 \lVert \Delta x \rVert_1 / \lVert x \rVert_1 \) falls below
`stop_threshold_change_pct=0.2`.  Here \( \Delta x \) is the change of the
image in one iteration.  With these defaults, the fixed code stops far from the
MAP estimate.  mbirtorch's default prior has \( q = 1.2 \) and \( T = 1 \).
With this prior at \( r = 0.1 \), 15 iterations moved the image from 155.8 to
69.6 HU from the MAP estimate.  The noise per pixel fell from 182 to 100 HU,
while the MAP estimate has 37 HU.

The stopping rule also depends on the image's offset.  Adding a constant to
\( y \) adds the same constant to the MAP estimate, because the prior depends
only on differences of neighbors.  However, the constant changes
\( \lVert x \rVert_1 \), and so it changes the iteration at which the rule
stops **[derived]**.

In a MACE or plug-and-play loop, a call of `denoise` can start from the
previous output, through `init_image`.  Then a few iterations per call may be
enough **[inferred]**.  A call that denoises an image once needs many more.

## A faster solver

L-BFGS reached the MAP estimate much faster than the fixed solver, which is
the vectorized coordinate descent (VCD) of `denoise`.  The table gives the
iterations and the time that each solver took to come within 0.01 HU of the
MAP estimate.  Each iteration count is the first at which a run in steps of
10 iterations for VCD, or of 1 for L-BFGS, came within 0.01 HU.  Each time is
the shorter of two uninterrupted runs to that count, on 4 CPU threads.  VCD
ran in float32, as in mbirtorch.  L-BFGS was `torch.optim.LBFGS` with a
history of 10 and the strong Wolfe line search.  It ran in float64 on
\( y / \sigma_y \).

| Prior | \( r \) | VCD iterations | VCD time (s) | L-BFGS iterations | L-BFGS time (s) | Time ratio |
|---|---:|---:|---:|---:|---:|---:|
| \( q = 1.2 \), \( T = 1 \) | 0.20 | 110 | 1.95 | 22 | 0.24 | 8 |
| \( q = 1.2 \), \( T = 1 \) | 0.10 | 420 | 7.52 | 47 | 0.48 | 16 |
| \( q = 1.2 \), \( T = 1 \) | 0.05 | 1930 | 34.05 | 100 | 1.06 | 32 |
| \( q = 2 \) | 0.20 | 70 | 0.94 | 20 | 0.13 | 7 |
| \( q = 2 \) | 0.10 | 280 | 3.66 | 41 | 0.24 | 15 |

The time ratio doubled each time \( r \) halved.  This growth is expected.
For \( q = 2 \), the condition number of the Hessian of \( J \) grows as
\( 1/r^2 \) **[derived]**.  The number of iterations of coordinate descent
grows in proportion to the condition number.  The number of iterations of
conjugate gradients grows as its square root, and L-BFGS behaves similarly
**[inferred]**.  In addition, an L-BFGS iteration took less time than a VCD
iteration here: 6 to 11 ms against 13 to 18 ms.

These results have four limits:

- They come from one 2D slice on a CPU.  Runs on a GPU and in 3D are untested.
- L-BFGS ran in float64.  Its behavior in float32 is untested.
- L-BFGS needs more memory.  `torch.optim.LBFGS` keeps two image-sized vectors
  per history entry, plus about four more.  So a history of 10 needs about 24
  copies of the image, and VCD needs about 4 **[derived]**.  For a large
  volume on a GPU, a short history may be needed.  Nonlinear conjugate
  gradients is another option, with about 5 copies **[derived]**.
- At \( r = 0.05 \), VCD in float32 stopped improving at about 0.008 HU from
  the MAP estimate.

## What to do

Take these steps in order, and report to Greg at the end of each step:

1. **Reproduce the scaling problem in the current checkout.**  A direct test
   is scale equivariance.  For each \( c \) in \( 10^{-3} \), 1 and
   \( 10^3 \), denoise \( c y \) with `sigma_noise` equal to
   \( c \sigma_y \) and `sigma_x` equal to \( c \sigma_x \).  Run a fixed
   number of iterations, with `stop_threshold_change_pct=0`.  Each result
   should equal \( c \) times the result for \( c = 1 \).  The released code
   should fail this test.  Run the test with the automatic `sigma_x` as well.
2. **Fix the three functions**, after Greg approves your plan.  The fixed code
   should pass the test at every \( c \).
3. **Check the tests that use the denoiser.**  These are
   `tests/test_denoiser.py`, `tests/test_mace.py` and `tests/test_mace4d.py`.
   At 63207f7, `tests/test_denoiser.py` stores no golden values, and its
   checks are relative.  However, the 0.0.2 docstring of
   `vcd_subset_denoiser` says that golden-value tests fix its formulas.  Find
   out whether any test, or any comparison with mbirjax, depends on the old
   update.
4. **Evaluate the stopping rule and the default number of iterations**, for
   each place that calls `denoise`.  Standalone denoising and the MACE agents
   may need different defaults.  A rule that compares the change per
   iteration with \( \sigma_y \) would not depend on the image's scale or
   offset.
5. **Evaluate a faster solver on a GPU and in 3D**, including its memory.
   Start from L-BFGS, and consider alternatives.  Do not replace VCD before
   Greg decides.
6. **Evaluate whether the clamp on alpha to 1.5 should be removed or whether the 
   tomography models need a similar scaling adjustment.**  The fix above addresses
   the scaling issues for the denoiser only.  The original intent of alpha was to limit steps
   estimated from a linear approximation but applied to a nonlinear problem, but it implicitly
   assumed unitless steps, so that alpha near 1 would be appropriate.  Determine whether similar scaling
   issues could arise for tomography problems and whether additional scaling is appropriate or
   the bound removed from alpha or a warning printed when alpha is far above 1.5. 

## Files in pcdrecon

The pcdrecon checkout is at
`/Users/gbuzzard/Documents/PyCharm Projects/Research/pcdrecon`, on branch
`greg_dev`.  The test is in `plans/image_mace/`:

- `plan.md` describes the test.  Its "Solver" section and its checks 1 to 5
  concern `denoise`.
- `experiments/qggmrf_map.py` computes \( J \) in torch for an image of shape
  (1, m, n).  It also has the L-BFGS solver used above, and the exact MAP
  estimate for \( q = 2 \), computed with the DCT.
- `experiments/check_solver.py` makes the synthetic slice and compares
  `qggmrf_map` with `denoise`.

The `results/` folder of the test holds GE data.  Do not copy anything from it
into mbirtorch or its tests.
