# Choosing beam-hardening correction parameters from image quality: a literature survey

Prepared 2026-09-05. Every citation below was checked against a record I actually
fetched: Europe PMC's REST API (which returns publisher abstracts), the Semantic
Scholar graph API, OSTI, PMC full text, or the project's own repository files.
Where I could only read an abstract and not the full paper, I say so at that point.

Two corrections to the leads I was given. The 2003 Elbakri and Fessler paper is in
Physics in Medicine and Biology, not IEEE TMI. Empirical binary tomography
calibration is by Grimmer and Kachelriess, not Kyriakou.

---

## 1. The classical two-material baseline

**Joseph and Spital, J. Comput. Assist. Tomogr. 2(1):100-108, 1978.** PMID 670461,
record at https://journals.lww.com/jcat/abstract/1978/01000/a_method_for_correcting_bone_induced_artifacts_in.17.aspx
This is the ancestor of the code in `mar.py`. Reconstruct once, segment bone, forward
project the bone class, and correct each line integral with a small polynomial in the
water path and the bone path. There is no image-domain objective; the correction comes
from an assumed known spectrum. I read only search-result summaries of this paper.

## 2. The empirical family (Kachelriess group)

**Kachelriess, Sourbelle, Kalender, Med. Phys. 33(5):1269-1274, 2006, DOI
10.1118/1.2188076 (ECC).** A polynomial precorrection applied to the raw data. The
coefficients are found in the image domain by fitting a series of basis images to a
template image. The template comes from the uncorrected phantom image itself, so no
assumption is made about phantom size or position. Needs: one calibration scan of a
homogeneous phantom. Needs neither the spectrum nor attenuation coefficients. Cost:
one calibration, then a lookup. Limits stated by the authors: this is a first-order
correction and does not compete with iterative higher-order beam-hardening or scatter
methods; they nonetheless report reduced bone-induced artifacts in mouse images. The
key structural idea for Greg is already here: **basis images, reconstructed once,
combined linearly, with coefficients solved in the image domain.**

**Kyriakou, Meyer, Prell, Kachelriess, Med. Phys. 37(10):5179-5187, 2010, DOI
10.1118/1.3477088 (EBHC).** Segment the reconstruction into materials. Monochromatically
forward project each class. Combine the measured raw data and the class raw data
monomially (products and squares). Reconstruct each monomial combination to give a set
of correction volumes. Add a weighted sum of these volumes to the original volume. The
weights are chosen "to maximize the flatness of the new and corrected volume" (OSTI
abstract of Kyriakou et al. 2010, https://www.osti.gov/biblio/22096777). Secondary
descriptions say the flatness functional is total variation of the combined image; I
could not read the full text, so treat the exact functional as unconfirmed. Parameter
count: for the standard water-plus-dense-material second-order case the abstract text
describes two extra volumes, hence two weights. Needs: a segmentation only; no
spectrum, no attenuation tables, no calibration phantom. Cost: one forward projection
per class plus one reconstruction per basis volume, then a low-dimensional search.
Limits: it depends on the segmentation, and the abstract notes that both polychromaticity
and scatter cause the underestimation it corrects, so the two are not separated.

**Schüller, Sawall, Stannigel, Hülsbusch, Ulrici, Hell, Kachelrieß, Med. Phys.
42(2):794-803, 2015, DOI 10.1118/1.4903281 (sfEBHC).** Replaces the segmentation with a
nonlinear histogram deformation that accentuates high CT values. The original volume
and the deformed volume are both forward projected, the two projection sets are
combined monomially, and the resulting basis volumes are combined by the same
maximization of image flatness. The authors state that no additional calibration or
parameter fitting is required beyond that. Reported to equal or beat EBHC on
simulations, phantoms, and patients, including noisy and strongly corrupted images.
Relevance for Greg: this removes the segmentation dependence but keeps the
basis-images-plus-flatness structure exactly.

Two siblings matter only as evidence that these empirical fits absorb scatter as well as
hardening. **Grimmer and Kachelriess, Med. Phys. 38:2233-2240, 2011, DOI
10.1118/1.3561506 (EBTC)** models both in one ray-specific calibration and reports about
97 percent hardening reduction and about 75 percent scatter reduction in simulation.
**Stenner, Berkus, Kachelriess, Med. Phys. 34(9):3630-3641, 2007, DOI 10.1118/1.2769104
(EDEC)** fits polynomial dual-energy decomposition functions by least squares on a
calibration phantom, with no spectrum, and reports that the fit inherently compensates
scatter.

## 3. The automatic image-quality family (Wilson group) — closest to the idea

**Levi, Fahmi, Eck, Fares, Wu, Vembar, Dhanantwari, Bezerra, Wilson, Proc. SPIE 9784,
97843S, 2016, DOI 10.1117/12.2216623**, and the journal version **Levi et al., Med. Phys.
46(4):1648-1662, 2019, DOI 10.1002/mp.13402 (ABHC-1)**. Two parameters in a second-order
correction on the projection of the high-attenuating class. The cost is flatness plus an
energy term with weight 0.61. Flatness is total variation of a gradient image
thresholded so the optimizer does not flatten the high-attenuating material itself. The
energy term is the absolute change in summed gray values relative to the
initial-parameter image. They report several local minima. Streaks fell from 13 plus or
minus 2 HU to 0 plus or minus 1 HU in a digital phantom and from 48 plus or minus 6 HU
to 1 plus or minus 5 HU in a physical phantom.

**Levi, Wu, Eck, Fahmi, Vembar, Dhanantwari, Fares, Bezerra, Wilson, Med. Phys.
48(1):287-299, 2021, DOI 10.1002/mp.14599 (ABHC-1/2/3 comparison).** The single most
useful paper for Greg's question; I read the full text at PMC8022227. They derive the
correction as a series expansion of a dual-material projection, reconstruct each cross
term as its own basis image, and add a weighted sum to the uncorrected reconstruction.
ABHC-2 has three free coefficients for two materials. ABHC-3 extends the expansion to
three materials with seven free coefficients. The cost is a convex combination of two
normalized image-domain terms: a gradient-based total variation inside the myocardium
divided by the myocardium area, plus a squared deviation from an expected value inside
the left ventricle divided by the ventricle area. A variant, ABHC-NH, uses plain total
variation of the whole corrected image with the same seven parameters and a simplex
optimizer. Segmentation is by soft thresholds, with mixed pixels treated as linear
combinations. Findings that matter: ABHC-2 beat ABHC-3 and ABHC-NH; ABHC-2 held
artifacts under 1.8 HU even at elevated noise; the seven-parameter variants were worse
than uncorrected FBP for bias and precision on noisy images; restarting ABHC-3 from many
initial points did not help. The authors attribute this to reduced estimability, and say
more parameters make the cost surface more complex and add local minima, which noise
makes worse. Runtime is not reported.

**Byl, Klein, Sawall, Heinze, Schlemmer, Kachelrieß, Med. Phys. 48:3572-3582, 2021, DOI
10.1002/mp.14931 (PCNMAR).** Photon-counting metal artifact reduction that builds a
prior by multiplying forward-projected bin sinograms with bone-emphasized sinograms to
mimic hardening, reconstructing those products, and linearly combining them with
coefficients found automatically by minimizing a threshold-based cost function in the
image domain. This is the EBHC construction carried into MAR.

## 4. Model-based polyenergetic reconstruction

These estimate the image itself under a polychromatic forward model, so the parameter
count is the voxel count and there is no small coefficient vector to pick by an image
criterion. **Elbakri and Fessler, IEEE TMI 21(2):89-99, 2002, DOI 10.1109/42.993128**
makes each voxel's attenuation an unknown density times a known mass attenuation curve,
with non-overlapping materials, solved by monotone penalized-likelihood ordered subsets;
it needs a segmentation, a known spectrum, and NIST-style tables. **Elbakri and Fessler,
Phys. Med. Biol. 48:2453-2477, 2003, DOI 10.1088/0031-9155/48/15/314** removes the
pre-segmentation and allows mixed pixels by writing the attenuation as an unknown density
times a weighted sum of known mass attenuation curves. **O'Sullivan and Benac, IEEE TMI
26:283-297, 2007, DOI 10.1109/TMI.2006.886806** recasts the polyenergetic maximum
likelihood problem, with background and a non-ideal point spread function, as a double
minimization of I-divergence; every alternating step is closed form and the objective
decreases monotonically. All three need the spectrum.

**Zhao and Li, PLoS ONE, 2015, DOI 10.1371/journal.pone.0144607.** Iterative BHC for
multi-material objects, solved as a nonlinear system with an extended ART, tested with
water, bone, and titanium. Requires known materials, an estimated spectrum, and NIST
tables. Reported robust to spectrum error but limited by: one material per voxel,
limited density variation, an assumption that scatter is negligible, residual radial
artifacts from quantum noise, and streaks at tangent lines to strong absorbers.
Converged in 6 to 13 iterations.

## 5. Blind and calibration-free methods

**Van Gompel, Van Slambrouck, Defrise, Batenburg, de Mey, Sijbers, Nuyts, Med. Phys.
38(S1):S36-S49, 2011, DOI 10.1118/1.3577758.** Segment into materials, parametrize the
spectrum with a few energy bins, then estimate the spectrum parameters and the material
attenuation values by minimizing the difference between measured and simulated
polychromatic sinograms. Three algorithms result: two reconstruction variants and one
sinogram precorrection. Reported to remove cupping and keep segmented regions
homogeneous even when the segmentation is poor. I read only the abstract, so I cannot
report their identifiability discussion first hand.

**Abdurahman, Frysch, Bismark, Melnik, Beuing, Rose, IEEE TMI 37(10):2266-2277, 2018,
DOI 10.1109/TMI.2018.2840343.** Estimates the polynomial correction coefficients by
minimizing the inconsistency of projection pairs under Grangeat's relation. Needs no
calibration, no spectrum, no attenuation properties, and no detector response. Reported
robust to measurement and geometric errors. This is the main alternative to an
image-domain criterion: the same parameter vector, a projection-domain objective.

**Zhao, Li, Niu, Qin, Peng, Niu, arXiv:1812.02365, 2018.** Estimates the spectrum, then
reprojects segmented template images. Multi-material by construction.

**Lifton, J. X-ray Sci. Technol. 25:629-640, 2017, DOI 10.3233/XST-16197.** A
multi-material linearization that runs like a mono-material one, about 0.02 s per
projection, and needs measured attenuation of one constituent material. He frames the
standard multi-material alternative as an iterative segmentation-based algorithm that is
expensive and can fail on a poor initial segmentation. Cupping in steel, titanium, and
aluminium spheres fell from 22, 20, 20 percent to 5, 1, 0 percent.

## 6. Metal artifact reduction and deep learning

The mainstream MAR line inpaints rather than fits a hardening polynomial, so it is
adjacent rather than competing: NMAR (Meyer, Raupach, Lell, Schmidt, Kachelriess, Med.
Phys. 37:5482-5493, 2010, DOI 10.1118/1.3484090) normalizes the sinogram by a prior's
forward projection before inpainting; FSMAR (Meyer et al., Med. Phys. 39:1904-1916,
2012, DOI 10.1118/1.3691902) restores high frequencies and needs no manual parameters;
NLS-NMAR (Anhaus, Killermann, Mahnken, Hofmann, Med. Phys. 50:4721-4733, 2023, DOI
10.1002/mp.16461) suppresses low-frequency inpainting artifacts. On deep learning,
Kleber et al., Eur. J. Radiol. 181:111732, 2024, DOI 10.1016/j.ejrad.2024.111732
reviewed fourteen supervised MAR studies and concluded that results are promising but
standardized clinical evaluation is missing. Nothing in that review selects a physical
correction parameter by an image criterion.

## 7. LEAP and XrayPhysics

LEAP (https://github.com/LLNL/LEAP) applies BHC through the XrayPhysics package
(https://github.com/kylechampley/XrayPhysics). The XrayPhysics README states it provides
one- and two-material beam hardening correction, in both a theoretically exact and a
polynomial form, over 1 keV to 20 MeV and elements 1 to 100, driven by hard-coded EPDL97
cross sections. Inputs are a chemical formula or mass fractions plus a source model. The
LEAP demo `demo_leapctype/d18_multi-materialBHC.py` iterates five times: single-material
BHC, reconstruct, threshold into low-Z / high-Z / mixed, compute the high-Z fraction,
then apply a dual-material lookup table. The limit is explicit in LEAP's own README:
support for **more than two materials** is listed as future work, alongside variable
takeoff angle and graded filtration. LEAP today is therefore a spectrum-driven,
table-driven, two-material method, and Greg's K-metal case is outside it.

---

## Question 1: closest prior method to reconstructing basis images once and combining them linearly under an image criterion

The closest is **ABHC-2 and ABHC-3 (Levi et al., Med. Phys. 48:287-299, 2021, DOI
10.1002/mp.14599)**, with EBHC (Kyriakou et al. 2010) and sfEBHC (Schüller et al. 2015)
as the direct ancestors and ECC (2006) as the origin of the basis-image trick.

The construction is identical to what Greg is considering. They expand the polychromatic
projection as a polynomial in the class path lengths, reconstruct one basis image per
cross term, and write the corrected image as the uncorrected image plus a weighted sum
of basis images. Only the scalar weights are searched, so the expensive projections and
reconstructions happen once.

Their criterion has two parts, both normalized by area. The first is a gradient-magnitude
total variation restricted to a homogeneous region (the myocardium), divided by that
region's pixel count. The second is a squared deviation from an expected value inside a
second region (the ventricle), divided by its area. The two are combined with a convex
weight. In the earlier SPIE version the second term was a total image energy term with
weight 0.61.

The pitfalls they report are exactly the ones to plan for:

- A pure flatness term will flatten the high-attenuating material itself. Their fix was
  to threshold the gradient image so the metal-like class is excluded, and to add a
  second term that anchors an absolute value. Without such an anchor, flatness alone is
  minimized by a constant image.
- The cost surface has several local minima, and more parameters make it worse.
- Estimability collapses between three and seven parameters. Three worked and was robust
  to noise; seven was worse than no correction at all on noisy images, and multiple
  restarts did not recover it.
- Parameters must be kept consistent across a series of related images or the corrected
  values fluctuate.

This last group is a direct warning for `mbirtorch`. In
`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/mbirtorch/preprocess/mar.py`,
`correct_sino_plastic_metal` builds `H_exponent_list` with `1 + num_cross_terms +
num_metal_terms` columns. For `num_metal=1, order=3` that is 6 coefficients; for
`num_metal=2` it is 15; for `num_metal=3` it is 29. The literature's only reported
image-criterion fit at seven parameters failed to be estimable under clinical noise. If
the image criterion is going to select the coefficients, the search space almost
certainly has to be reduced first, for example by keeping the sinogram least-squares fit
for most coefficients and letting the image criterion move only a few directions.

## Question 2: what the field says about more than two materials

Almost everything published and validated is two materials, usually soft tissue plus
bone, or plastic plus one metal. The three-material results are sparse and cautious.

- LEAP's own roadmap lists BHC for more than two materials as not yet implemented.
- ABHC-3 is the only image-criterion method I found that goes to three materials
  explicitly, and its authors report that it was not estimable at clinical noise.
- Zhao and Li (PLoS ONE 2015) do handle K materials and demonstrate K=3, but they need a
  spectrum estimate, NIST tables, one material per voxel, and negligible scatter.
- Lifton (2017) explicitly frames multi-material BHC as the case where the standard
  answer is an expensive iterative segmentation loop that can fail on a poor initial
  segmentation, and offers a linearization that needs measured attenuation of one
  constituent.
- Van Gompel et al. (2011) allow several materials but estimate spectrum bins and
  material attenuations jointly, which trades the parameter explosion in the polynomial
  for a parameter explosion in the physical model.

The consistent message: the number of cross terms grows combinatorially with the number
of classes, and no one has reported reliably fitting that many free parameters from
image quality alone. The methods that succeed with several materials all import physics
(a spectrum, cross sections, or a measured material) to pin most of the coefficients.

## Question 3: reported non-identifiability and degeneracy

Four separate degeneracies appear in the literature, and only some are named as such.

**Flatness versus scale.** A total variation or flatness criterion is invariant to adding
a constant and nearly invariant to scaling, so a flatness-only objective drives toward a
trivially flat image. Levi et al. name this directly: their gradient image is thresholded
so the correction does not flatten the high-attenuating material, and their second cost
term anchors an absolute level. ECC solves the same problem differently, by fitting basis
images to a template with a known target value.

**Spectrum versus material.** Sidky, Yu, Pan, Zou, Vannier, J. Appl. Phys. 97:124701,
2005, DOI 10.1063/1.1928312 state that spectrum estimation from transmission data is
ill-conditioned, and design the transmission measurements specifically to reduce that
instability. Duan, Wang, Yu, Leng, McCollough, Med. Phys. 38(2):993-997, 2011, DOI
10.1118/1.3547718 report a concrete symptom: their EM estimate cannot recover a
characteristic peak unless the initial guess already has a peak at that energy, so they
seed it with a tungsten spectrum times the detector response. A joint fit of spectrum and
material properties is therefore weakly determined, which argues for fitting the
composite polynomial rather than the physics behind it.

**Beam hardening versus scatter.** No empirical method separates them, and several authors
treat this as a feature. The EBHC abstract attributes the underestimation to both
polychromaticity and scatter. EDEC compensates scatter inherently. EBTC deliberately
models both in one calibration and quantifies both reductions. A coefficient set fitted by
an image criterion should be expected to absorb scatter, and its values should not be read
as spectral physics.

**Parameter count versus estimability.** Levi et al. 2021 is the cleanest published
statement: three parameters were robust and seven were not, in the same framework, on the
same data, with multiple restarts, which they tie to local minima amplified by noise. That
is not a degeneracy proof, but it is the strongest empirical evidence here that an image
criterion alone cannot support many coefficients.

## Question 4: five papers to read first, ranked

1. **Levi et al., Med. Phys. 48(1):287-299, 2021, DOI 10.1002/mp.14599.** It builds the
   exact thing Greg is proposing, with basis images combined linearly under an
   image-domain cost, and it reports where the parameter count breaks.
2. **Kyriakou, Meyer, Prell, Kachelriess, Med. Phys. 37(10):5179-5187, 2010, DOI
   10.1118/1.3477088.** The canonical method that reconstructs monomial correction
   volumes and picks the weights by image flatness, with no spectrum and no calibration.
3. **Kachelriess, Sourbelle, Kalender, Med. Phys. 33(5):1269-1274, 2006, DOI
   10.1118/1.2188076.** The origin of fitting basis images to a template in the image
   domain, and the clearest example of how to anchor the absolute scale that flatness
   cannot fix.
4. **Schüller et al., Med. Phys. 42(2):794-803, 2015, DOI 10.1118/1.4903281.** Shows how
   to keep the same basis-and-flatness machinery while removing the dependence on a hard
   segmentation, which matters when metals and plastic segment badly.
5. **Zhao and Li, PLoS ONE, 2015, DOI 10.1371/journal.pone.0144607.** The best-documented
   genuinely multi-material iterative correction, with an honest list of what breaks:
   mixed voxels, density variation, scatter, and tangent-line streaks.

Worth reading sixth, if the projection domain is on the table as an alternative
criterion: **Abdurahman et al., IEEE TMI 37(10):2266-2277, 2018, DOI
10.1109/TMI.2018.2840343**, which fits the same kind of polynomial by minimizing
projection-pair inconsistency rather than image roughness, with no priors at all.
