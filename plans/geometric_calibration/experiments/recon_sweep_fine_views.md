# Viewing the fine sweep's candidates on the phantom's features: what the two viewing scripts show

Date: 2026-09-05.  During his review of the fine-sweep findings Greg asked to see the candidate
rotations on the reconstructions themselves: the slices side by side, a profile through the
horizontal groove, and whether the vendor's angle makes the boundary notches less deep.  Two
scripts beside this page answer those questions from the Fourier-kernel job's saved stacks, and
this page records their numbers.  The scripts are `recon_sweep_fine_slice_views.py` and
`recon_sweep_fine_feature_profiles.py`.  They ran on a Mac against local copies of four stack
pairs of gautschi job 15971893 (`results_recon_sweep_fine_fourier`, scan `nsi_no_metal`, slices
1409 and 1691, even and odd views).  Every number below was read in this session from the
scripts' printed output, and the figures they wrote are ignored by this repository, so this page
is the durable copy.  Both scripts recomputed the jobs' own scores from the stacks and matched
the record to a relative 1.1e-09 or better.

## Terms

The candidates drawn are 0.130 degrees, the slice's own best grid candidate, 0.150 degrees, and
the vendor's 0.16717 degrees.  The scoring blur is the Gaussian of 2 reconstruction pixels the
sweep recommends.  The groove is the deep horizontal dark line across the disk on slice 1691.
The teeth are the boundary protrusions of the disk on slice 1409, and the gaps between them are
the notches.  A paired comparison evaluates the same pixels in two candidates' reconstructions,
so the phantom's own variation cancels and only the candidate difference remains.

## What the slice views show

`recon_sweep_fine_slice_views.py` draws each slice at the four candidates: the full slice with
the crop marked, the slice's score curve, zoomed crops on the object's boundary, and each crop's
difference from the best candidate after the scoring blur.  By eye the crops are
indistinguishable, because between these candidates a far slice's content moves by half a pixel
or less.  The blurred differences show the boundary displacement growing with distance from the
slice's own optimum, and the per-slice sharpness drops by 1 to 2 percent at the vendor's value.
The crop is restricted to the object's boundary because the strongest candidate difference lies
on the holder ring at the field's rim, which rotates with the stage but is not the object.

## The groove

A vertical line through the groove, five columns averaged, shows a dip to about a quarter of the
disk's level, and the four candidates' single-line profiles overlay within the noise.  Averaging
240 columns resolves them.  The table gives the depth below the shoulders and the width at half
depth of the averaged profile, per candidate, from the script's output.

| candidate, degrees | depth, raw | width, raw, rows | depth, blurred | width, blurred, rows |
| --- | --- | --- | --- | --- |
| 0.1300 | 0.013364 | 4.47 | 0.009072 | 6.61 |
| 0.1350 (slice best) | 0.013433 | 4.45 | 0.009086 | 6.59 |
| 0.1500 | 0.013488 | 4.43 | 0.009093 | 6.59 |
| 0.1672 (vendor) | 0.013296 | 4.48 | 0.009037 | 6.62 |

The paired comparison over 24 blocks of 10 columns, on the blurred profiles, puts the vendor's
groove 0.000060 shallower than the slice-best candidate's, with a standard error of 0.000009.
That is about 0.7 percent of the blurred depth, at roughly seven standard errors, and the vendor's
groove is also the widest.  The 0.150-degree groove is marginally the deepest.  These results
confirm the direction of Greg's visual impression on the feature he meant: the vendor's angle
does give a less deep groove, by less than a percent.

## The notch modulation

`recon_sweep_fine_feature_profiles.py` also samples slice 1409 around the circle through
mid-tooth height, radius 324.8 pixels, where the trace alternates between tooth material and gap
air: 362 tooth samples and 1779 gap samples, with the arcs fixed from the sharpest candidate and
the rays near the groove excluded.  The modulation depth is the tooth mean minus the gap mean.

| candidate, degrees | modulation depth | gap level |
| --- | --- | --- |
| 0.1250 (slice best) | 0.012778 | 0.004473 |
| 0.1300 | 0.012781 | 0.004474 |
| 0.1500 | 0.012776 | 0.004485 |
| 0.1672 (vendor) | 0.012750 | 0.004499 |

The paired comparison over the 32 gaps puts the vendor's gaps 0.000026 brighter than the
slice-best candidate's, with a standard error of 0.000009, and the fill grows monotonically with
the distance from the slice's optimum.  The vendor's angle therefore fills the notches by about
0.2 percent of the modulation, a real effect at three standard errors and far below what one
notch can show against its own noise.

## Reading

Three points follow.  A single feature at these candidate separations is noise-limited, so a
visual impression from one notch or one line cannot be trusted at the sub-percent level; the
paired averages are what resolve it.  Every per-feature ordering here matches the score's
ordering on the same slice, and the per-slice optima differ between the two slices, which is the
offset-term signature the sweep's record describes.  These are viewing aids beside the sweep's
gates, not gates themselves.

## Limits, and how to run

Four limits apply.  One scan, two slices, and one blur setting were drawn.  The features were
measured on the Fourier job's stacks only.  The windows and arcs were fixed from the sharpest
candidate, so a different reference could move the numbers slightly.  The scripts ran on a Mac
against copied stacks rather than in a recorded cluster job.

Both scripts read the results directory from the environment variable `RECON_SWEEP_FINE_RESULTS`
and the stacks directory from `RECON_SWEEP_FINE_STACKS`, which defaults to the results directory,
as on the cluster where the stacks sit beside the record.  They need numpy, scipy, and
matplotlib, and no GPU.
