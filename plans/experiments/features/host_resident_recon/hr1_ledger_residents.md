# hr1: the loop's device residents, priced by the memory ledger

## What the script does

The script builds a cone-beam model for three geometries and asks the memory
ledger, with and without a proximal input, for three quantities of the
reconstruction loop: the persistent set, which is every array the loop holds
for its whole run; the widest subset step, which is the per-device peak the
device policy fits to memory; and the recon-shaped residents, which a
host-resident layout would remove from the device.  It also reports how many
pixels the partitions cover, and the dominant phase of the whole run.  No
array is allocated; the ledger prices from the shapes.

## Command

```
cd "/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch" && \
/Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python \
  "/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_plans/plans/experiments/features/host_resident_recon/hr1_ledger_residents.py" \
  2>&1 | grep -v Warning | tee ".../results/hr1_ledger_residents.txt"
```

## Environment

Python 3.11.15, torch 2.13.0, numpy 2.4.6, on the Mac with a CPU layout.
The ledger prices the projection bodies the model would run on the pricing
device.  On this Mac those are the torch bodies.  Their per-step transients
are far larger than the Triton kernels' on an H100: the forward batch is
charged at 28 GB in every case, the back batch at 88 GB in the loop's back
projection step and at 352 GB in the pre-loop phases.  So the widest-step
totals and the dominant-phase totals below overstate a GPU run's peak, and
the pre-loop phases dominate the run only because of those terms.  The
persistent terms depend on the geometry only.

## The persistent set and the widest subset step, in gigabytes

Weights supplied in every case.  Dense means 3072 views of 2048 by 2048 on
a 2048-cubed volume.  Sparse means 512 views on the same volume.  The
partitions cover 3,290,904 of the 4,194,304 pixels of the grid, a fraction
of 0.7846, because the default region-of-reconstruction mask leaves the
corners out.

| Quantity | Dense, qGGMRF | Dense, prox | Sparse, qGGMRF | Sparse, prox |
|---|---|---|---|---|
| error sinogram | 48.00 | 48.00 | 8.00 | 8.00 |
| weights | 48.00 | 48.00 | 8.00 | 8.00 |
| flat recon | 32.00 | 32.00 | 32.00 | 32.00 |
| hessian diagonal | 32.00 | 32.00 | 32.00 | 32.00 |
| prox input | - | 32.00 | - | 32.00 |
| partitions and workspace | 0.35 | 0.35 | 0.35 | 0.35 |
| persistent set | 160.36 | 192.36 | 80.36 | 112.36 |
| recon-shaped residents | 64.00 | 96.00 | 64.00 | 96.00 |
| recon-shaped share of the persistent set | 39.9% | 49.9% | 79.6% | 85.4% |
| widest subset step (back projection, granularity 4) | 327.62 | 359.62 | 207.62 | 239.62 |
| of which torch-body back batch | 87.88 | 87.88 | 87.88 | 87.88 |
| recon-shaped share of the widest step | 19.5% | 26.7% | 30.8% | 40.1% |

The other terms of the widest step are the prior gradient and Hessian at
12.55 GB, the weighted error sinogram at the sinogram's size, and the back
output at 18.83 GB.  The 512-class dense case has a persistent set of
2.58 GB without a prox input and 3.08 GB with one, and the same shares of
the persistent set to within one percent.

## What the numbers say

The recon-shaped residents are two full volumes, or three with a proximal
input.  Their share of the persistent set is 40 percent for the dense
qGGMRF case, 50 percent with a prox input, and 80 and 85 percent for the
sparse cases.

Their share of the widest subset step is smaller, because that step also
holds the back projection's transients.  As priced here it is 20, 27, 31,
and 40 percent.  The torch-body back batch is 88 GB of every widest-step
total, so on an H100 the share lies between the two figures for each case.
Only a ledger run on the H100 settles it.

The dominant phase of the whole run is a pre-loop phase in every 2048-class
case, the direct reconstruction or the Hessian diagonal, because those
phases carry a 352 GB torch-body back batch here.  Without that term the
loop's back projection step is the widest phase for the dense cases and
the loop's prior step for the sparse cases.

Nothing failed and nothing was skipped.
