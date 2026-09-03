# LEAP (LivermorE AI Projector for Computed Tomography) — feature and performance inventory

Prepared for a comparison against mbirtorch. This document covers only the LEAP side.

## 0. What was inventoried

- Clone: `git clone --depth 1 https://github.com/LLNL/leap` into a temporary directory, read-only;
  nothing was built or installed.  That temporary clone is gone.  Every source reference below was
  rewritten to a permanent GitHub URL pinned to the commit that was read, so each one still resolves:
  https://github.com/LLNL/LEAP/tree/0c8846f42b2e59340d5559fc1271d590a292f9a0
- Clone HEAD is commit `0c8846f42b2e59340d5559fc1271d590a292f9a0`, dated 2024-12-14, commit message "updated version" (`git log -1`).
- That commit is also tag `v1.26` (`git ls-remote --tags https://github.com/LLNL/leap`).
- Version string in the source: `#define LEAP_VERSION "1.26"` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.h#L16), and `version='1.26'` in https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/setup.py, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/setup_ctype.py, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/setup_AMD.py.
- The wiki was cloned separately from `https://github.com/LLNL/LEAP.wiki.git`.  Wiki references below
  are links to the pages at https://github.com/LLNL/LEAP/wiki , which is not version-pinned.
- Every reference below of the form `https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/<path>#L<n>`
  names a file and line in the LEAP repository at that commit.  Line ranges appear as `#L<a>-L<b>`.

---

## 1. Project identity, license, funding

- Full name: "LivermorE AI Projector for Computed Tomography (LEAP)" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/README.md).
- Description on GitHub: "comprehensive library of 3D transmission Computed Tomography (CT) algorithms with Python and C++ APIs, a PyQt GUI, and fully integrated with PyTorch" (GitHub API `repos/LLNL/LEAP`, field `description`).
- Language reported by GitHub: `Cuda` (GitHub API `repos/LLNL/LEAP`, field `language`).
- Authors listed: Kyle Champley (champley@gmail.com) and Hyojin Kim (hkim@llnl.gov) (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/README.md).
- License: MIT. "SPDX-License-Identifier: MIT / LLNL-CODE-848657" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/README.md, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/LICENSE). Copyright line: "Copyright (c) 2013-2023 LLNS, LLC and other LEAP Project Developers." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/LICENSE).
- Funding: "This work was produced under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/NOTICE).
- Documentation site: https://leapct.readthedocs.io (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/README.md; also the `homepage` field of the GitHub API record).
- Separate PyQt GUI project: https://github.com/kylechampley/LEAPCT-UI-GUI (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/README.md).
- Physics companion package: https://github.com/kylechampley/XrayPhysics (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/README.md).

---

## 2. Geometries

### 2.1 Geometry types supported

The C++ enumeration of geometry types is the authoritative list:

- `enum geometry_list { CONE = 0, PARALLEL = 1, FAN = 2, MODULAR = 3, CONE_PARALLEL = 4 };` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/parameters.h#L620).

Python setters, one per type (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py, and listed in https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/ctgeometries.rst):

- `set_parallelbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L593).
- `set_fanbeam(..., phis, sod, sdd, tau=0.0)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L545).
- `set_conebeam(..., phis, sod, sdd, tau=0.0, helicalPitch=0.0, tiltAngle=0.0)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L479).
- `set_coneparallel(..., phis, sod, sdd, tau=0.0, helicalPitch=0.0)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L431). The demo notes "Cone-parallel coordinates is the standard coordinate system used in medical CT." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d10_fan_to_parallel_rebinning_and_cone-parallel.py).
- `set_modularbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions, moduleCenters, rowVectors, colVectors)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L638). Each of `sourcePositions`, `moduleCenters`, `rowVectors`, `colVectors` is an (numAngles x 3) array, i.e. an arbitrary source position, detector-module centre, and detector orientation per view.

### 2.2 Detector shape

- `enum detectorType_list { FLAT = 0, CURVED = 1 };` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/parameters.h#L622).
- `set_flatDetector()` — "Set the detectorType to FLAT" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L818).
- `set_curvedDetector()` — "Set the detectorType to CURVED (only for cone-beam data)" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L824). Flat is the default: "To switch between flat and curved detectors, use the set_flatDetector() and set_curvedDetector() functions. Flat detectors are the default setting." (`set_conebeam` docstring, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py).
- A curved detector made of discrete flat modules on an arc is handled by rebinning: `rebin_curved(g, fanAngles, order=6)` — "Real curved detectors are composed of a series of detector modules curved around a polygon shape. There are often gaps between modules that need to be accounted for and that is what this function does." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1239, demo https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d09_curved_polygon_detector_array.py).

### 2.3 Helical scans

- `set_helicalPitch(helicalPitch)` in mm/radians, "(cone-beam and cone-parallel data only)" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L764).
- `set_normalizedHelicalPitch(normalizedHelicalPitch)` with the stated conversion `h = numRows * pixelHeight * (sod/sdd) / (2*pi) * hHat` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L788).
- Helical FBP is GPU-only: "This script is nearly identical to d01_standard_geometries.py except it demonstrates LEAP's helical cone-beam functionality. LEAP has an implementation of helical FBP for the GPU only" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d03_helical.py); the C++ prints "Error: CPU-based FBP not yet implemented for helical cone-beam geometry" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/filtered_backprojection.cpp#L441).

### 2.4 Detector and source misalignments

- `tau` — "center of rotation offset in mm (fan- and cone-beam data only)" (`set_tau`, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L736).
- `tiltAngle` — "the rotation of the detector around the optical axis (degrees)" (`set_tiltAngle`, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L750); a cone-beam parameter.
- `centerRow` / `centerCol` — detector shift expressed as the pixel index the optical-axis ray hits (`set_centerCol` at https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L839, `set_centerRow` at https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1069).
- `shift_detector(r, c)` — "Shifts the detector by r mm in the row direction and c mm in the column direction by updating the CT geometry parameters accordingly" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1232).
- `rotate_detector(alpha)` — modular-beam only; scalar rotates around the optical axis, or a 3x3 rotation matrix can be supplied (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1116).
- `rotate_coordinate_system(R)` — "This main purpose of this algorithm is to enable reconstruction on arbitrary voxel grid. This is only possible for modular-beam geometries... Note that if after the rotation, the vector pointing along across the detector rows isn't aligned within 5 degrees of the z-axis, then one will not be able to perform FBP reconstructions" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1188).
- Non-uniform angular spacing: `phis` is an arbitrary float32 array of angles in degrees for every geometry setter; `setAngleArray(numAngles, angularRange)` is a convenience for equispaced angles (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1895). The feature list states "non-uniform angular spacing" explicitly (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/LEAP_features.md, item 5).

### 2.5 Conversions between geometries

- `convert_to_modularbeam()` — "Converts parallel- or cone-beam data to a modular-beam format for extra customization of the scanning geometry" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1090); also `convert_conebeam_to_modularbeam()` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1104) and `convert_parallelbeam_to_modularbeam()` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1110).
- `rebin_parallel(g, order=6)` — "rebin data from fan-beam to parallel-beam or cone-beam to cone-parallel" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1270). "Currently, the rebinning routines are only implemented on the CPU. If you require the rebinning to take place on the GPU please submit a feature request." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d10_fan_to_parallel_rebinning_and_cone-parallel.py).
- `rebin_parallel_sinogram(g, order=6, iRow=-1)` — single sinogram, for geometric calibration (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1300).

### 2.6 2D versus 3D

- Everything is stored as a 3D array. "Two-dimensional geometries can be achieved by setting the number of detector rows and number of z-slices to one." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex, section "CT Geometries").
- `FBP_slice(g, islice=None, coord='z')` reconstructs a single slice along `'x'`, `'y'`, or `'z'` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2504).

### 2.7 Special-purpose geometries and object models

- **Laminography** — no dedicated geometry; done with the modular-beam type. "This demo script simulates and reconstructs cone-beam laminography data... For this we will use the modular-beam geometry." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d32_laminography.py).
- **Offset scan / half-fan / half-cone** — `set_offsetScan(True)`, "This is sometimes refered to as a half-fan or half-cone or half-scan... it enables one to nearly double the diameter of the field of view" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5739; demo `d23_offsetScan.py`). The manual notes it "requires projections over 360 degrees" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex, table "Special purpose FBP parameters/algorithms").
- **Truncated scan** — `set_truncatedScan(True)`, uses "extrapolation of the signal instead of zero-padding when applying the ramp filter" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5721; demo `d22_truncatedScan.py`).
- **Cylindrically symmetric objects (Abel-transform-like)** — `set_axisOfSymmetry(val)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5767). Activation condition in C++: `numAngles == 1 && fabs(axisOfSymmetry) <= 30.0 && (geometry == CONE || geometry == PARALLEL)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/parameters.cpp#L627). "Because the object is symmetric, all reconstructions are essentially 2D because numX = 1" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d07_symmetric_objects.py). Motivated by flash x-ray radiography.
- **Attenuated Radon Transform (ART)** — `set_attenuationMap(mu)` for a voxelized map or `set_cylindircalAttenuationMap(c, R)` for a cylinder (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5876, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5910). "Note that only parallel-beam geometries are supported... We provide analytic reconstruction (FBP) of the ART for either specification of the attenuation map via Novikov's inversion formula. Unfortunately we only have this implemented for 360 degree angular range." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d06_ART.py). Backprojection: "Error: attenuated backprojection only works for parallel-beam data!" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors_attenuated.cu#L878, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors_attenuated.cu#L990). Applications named: SPECT and Volumetric Additive Manufacturing (VAM).
- **Circle-plus-line trajectory** — no analytic algorithm. "Note that LEAP does not have an implementation of the analytic circle+line algorithm. We solve this problem here by performing an axial FBP (FDK) reconstruction followed by an iterative reconstruction refinement using the line scan." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d24_circle_plus_line.py).
- **Detector dithering** — random per-view detector or stage shifts, modelled with modular-beam (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d25_detector_dithering.py).

### 2.8 Coordinate and detector conventions

- Detector sample positions: `s[i] = pixelWidth*(i - centerCol)`, `t[j] = pixelHeight*(j - centerRow)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/ctgeometries.rst; https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex).
- "For a perfectly centered detector, centerCol = 0.5(numCols-1) and centerRow = 0.5(numRows-1)" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex).
- "The origin of the coordinate system is always at the center of rotation." (each geometry docstring, e.g. `set_conebeam`).
- Right-handed system with z along the detector row coordinate and x along the detector column coordinate (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex, section "A Note on Coordinate Systems").
- Distances in mm; "The units of the voxelized volume (i.e., the reconstruction volume) are assumed to be in inverse length, e.g., mm^-1 or cm^-1." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex, section "Volume Parameterization").
- Explicit forward and adjoint integral formulas are given in the docstrings for parallel, fan, cone (flat and curved), cone-parallel, and modular beam (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py, and reproduced in https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/ctgeometries.rst).

---

## 3. Volume specification

- `set_volume(numX, numY, numZ, voxelWidth=None, voxelHeight=None, offsetX=None, offsetY=None, offsetZ=None)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1565). Sample positions:
  `x[i] = voxelWidth*(i-(numX-1)/2) + offsetX`, `y[j] = voxelWidth*(j-(numY-1)/2) + offsetY`, `z[k] = voxelHeight*(k-(numZ-1)/2) + offsetZ`.
- Voxels are square in x and y (one `voxelWidth`) but the z pitch is independent (`voxelHeight`). Arbitrary volume offsets in all three axes.
- `set_default_volume(scale=1.0)` — "The default volume parameters are those that fill the field of view of the CT system and use the native voxel sizes." The `scale` argument scales the voxel size and is annotated "(not recommended for fast reconstruction)" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1608).
- `set_diameterFOV(d)` — diameter of the circular mask applied to each z-slice. "Applying this mask removes artifacts outside the field of view that can be distracting. It also provides speed improvements and for cone-beam geometries can help algorithms to use less memory. If one does not want any masking applied, just provide a very large number" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5683).
- `windowFOV(f)` — zeroes voxels outside the field of view (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2940).
- `set_volumeDimensionOrder(which)` — 0 = XYZ, 1 = ZYX (the default). "WARNING: multi-GPU processing only works for ZYX order" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1630). The manual adds "The ZYX ordering works best for GPU processing (and is required for multi-GPU processing) and the XYZ ordering works best for CPU processing." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex).
- `set_volume_mask(vol_mask)` / `clear_volume_mask()` / `apply_volume_mask(f)` — a binary 3D mask of voxels to include. "Using this mask does not improve the speed of forward or backprojection algorithms. This mask is applied after performing a backprojection and before performing a projection." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3378).
- Region-of-interest support for chunked processing: `rowRangeNeededForBackprojection(iz=None)` and `sliceRangeNeededForProjection(doClip=True)` return the detector rows / z-slices actually needed, "which can be important to speed up calculations or reduce the CPU and/or GPU memory necessary to perform reconstruction" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2962, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3046).
- Voxel coordinate helpers: `x_samples()`, `y_samples()`, `z_samples()`, `voxelSamples()` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/ctvolume.rst).

---

## 4. Projector models

### 4.1 Available models

- `enum whichProjector_list {SIDDON=0, JOSEPH=1, SEPARABLE_FOOTPRINT=2, VOXEL_DRIVEN=3};` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/parameters.h#L623).
- The Python API exposes only two: `set_projector(which='SF')` accepts `'SF'` (modified Separable Footprint) or `'VD'` (Voxel-Driven) (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5788). The C++ layer rejects the others: "Error: currently only SF and VD projectors are implemented!" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L1930).
- Important asymmetry: "Note that all forward projectors use the modified separable footprint model, this function only changes the backprojection model. Voxel-driven backprojection is faster, but less accurate" (`set_projector` docstring, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5788).
- The Siddon CPU projector is present but marked deprecated: "C++ module for CPU Siddon projector (deprecated)" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors_Siddon_cpu.cpp#L7, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors_Siddon_cpu.h#L7).
- Joseph projectors exist and are used automatically for modular-beam geometry: on GPU, `project_Joseph_modular(...)` is selected when `params->geometry == parameters::MODULAR` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors.cpp#L78); the CPU path uses `project_Joseph_modular_cpu` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors.cpp#L120).
- An "extended SF" projector is selected automatically for voxels outside the fast-SF size range: `project_eSF` / `backproject_eSF` are entered when `params->voxelSizeWorksForFastSF(...) == false` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors_SF.cu#L2209, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors_SF.cu#L2329, implemented in https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors_extendedSF.cu).
- Separate specialized kernels: `projectors_symmetric.cu` (cylindrical symmetry) and `projectors_attenuated.cu` (Attenuated Radon Transform), selected before the general path (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors.cpp#L74-L79).

### 4.2 Dispatch logic (GPU path)

From https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors.cpp#L74-L80 (forward) and https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors.cpp#L160-L170 (backward), in order:

1. symmetric object -> `project_symmetric` / `backproject_symmetric`;
2. attenuation map specified -> `project_attenuated` / `backproject_attenuated`;
3. (backprojection only) `whichProjector == VOXEL_DRIVEN` -> `backproject_VD`;
4. `geometry == MODULAR` -> Joseph modular;
5. otherwise -> SF (which itself may fall through to extended SF).

### 4.3 When SF is used vs not

`parameters::useSF()` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/parameters.cpp#L614):

```
if (whichProjector == SIDDON || geometry == MODULAR || isSymmetric() == true)
    return false;
else
    return voxelSizeWorksForFastSF();
```

`voxelSizeWorksForFastSF(int whichDirection)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/parameters.cpp#L483) computes, for cone and modular geometries, the largest and smallest projected detector footprint of a voxel and returns false for backprojection when `voxelWidth > 2.0*smallestDetectorWidth || voxelHeight > 2.0*smallestDetectorHeight`, and false for forward projection when `0.5*largestDetectorWidth > voxelWidth || 0.5*largestDetectorHeight > voxelHeight`. In practice this means the fast SF kernels assume voxels roughly between 0.5x and 2x the nominal size, and the extended-SF kernels handle the rest.

### 4.4 Matched-pair and accuracy claims

- "Quantitatively accurate, matched (forward and back) projector pairs that model the finite size of the voxel and detector pixel; very similar to the Separable Footprint method [Long, Fessler, and Balter, TMI, 2010]. These matched projectors ensure convergence and provide accurate, smooth results. Unmatch projectors or those projectors that do not model the finite size of the voxel or detector pixel may produce artifacts when used over enough iterations [DeMan and Basu, PMB, 2004]." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/LEAP_features.md, item 2).
- "The projectors for parallel-, fan-, cone-, and axially aligned modular-beam geometries in LEAP are implemented by a method similar to the Separable Footprint (SF) method. These projectors are matched, meaning that the backprojection operation is an exact transpose of the forward projection and they model the finite size of the voxel and detector pixels. Our models are slightly faster than the original rectangle-rectangle SF method because they utilize the bilinear interpolation capabilities of texture memory, but are very, very slightly less accurate." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex, section "Matched Projector Models").
- "For this geometry [modular-beam], we use the projector method described above when the detector columns are roughly aligned with the z-axis and a matched Joseph projector [Joseph, TMI, 1982] otherwise." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex, Introduction).
- Note the tension worth flagging in a comparison: `set_projector('VD')` gives a backprojector that is *not* the transpose of the SF forward projector, so the matched-pair property holds only for the default `'SF'` setting.

### 4.5 Precision

- Float32 only. "The CPU- and GPU-based projectors are nearly identical (32-bit floating point precision)" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex, Introduction).
- The Python layer enforces it: "Error: projection and volume data must be float32 data type" and "must be contiguous" (`verify_inputs`, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L353).
- A grep for `__half`, `float16`, `half2`, `__nv_bfloat` across `src/*.cu`, `*.cuh`, `*.cpp`, `*.h` returned nothing: no half-precision support.
- No double-precision accumulation was found in https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors_SF.cu.
- CUDA is compiled with `--use_fast_math` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/CMakeLists.txt, `target_compile_options(leapct PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)`).
- Texture memory is used heavily: `cudaTextureObject_t` appears 16 times in `projectors_SF.cu`, 15 in `projectors_Joseph.cu`, 12 in `backprojectors_VD.cu`.

### 4.6 Projector API

- `project(g, f, param_id=None)` and `backproject(g, f, param_id=None)` — in-place into user-supplied arrays; return the same array (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1956, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2075).
- Explicit CPU/GPU variants: `project_cpu`, `project_gpu`, `backproject_cpu`, `backproject_gpu` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1993, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2035, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2101, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2131).
- `sensitivity(f=None)` — the backprojection of all-ones. "One can get the same result by backprojecting an array of projection data with all entries equal to one. The benefit of this function is that it is faster and uses less memory." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2910).
- Data layout (from https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex, "Layout of the Data Arrays"): projections are `projection_data[iAngle*numRows*numCols + iRow*numCols + iCol]`; volume ZYX is `volume_data[iZ*numY*numX + iY*numX + iX]`; XYZ is `volume_data[iX*numY*numZ + iY*numZ + iZ]`.
- Allocation helpers `allocate_projections(val=0.0, astensor=False, ...)` and `allocate_volume(val=0.0, astensor=False)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1712, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1812); `allocateProjections_gpu` / `allocateVolume_gpu` allocate torch tensors directly on the GPU (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1754, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1848).

---

## 5. Analytic reconstruction (FBP)

Functions listed in https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/fbp.rst:

- `FBP(g, f=None, inplace=False)` — "This function performs analytic reconstruction (i.e., FBP) of nearly all LEAP geometries: parallel-, fan-, cone-, and (axially-aligned) modular-beam geometries, including both flat and curved detectors, axial or helical scans, Attenuated Radon Transform, symmetric object, etc. Note that FDK is an FBP-type algorithm, so for simplicity we just called it FBP in LEAP." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2595). `inplace=True` filters `g` in place to save memory.
- `FBP_slice(g, islice=None, coord='z')` — single slice along x, y or z, without changing LEAP parameters (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2504).
- `FBP_adjoint(g, f)` — "This function will not provide an exact adjoint of FBP for fan-beam or helical scans." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2749).
- `BPF(g, f)` — Backprojection Filtration. "This reconstruction only works for parallel-beam data" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2887).
- `LT(g, f=None, inplace=False)` — Lambda / Local Tomography. "LT reconstructions work even when the projections are truncated and reconstruct the 2D ramp filtered volume which is essentially an edge map." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2797).
- `inconsistencyReconstruction(g, f=None, inplace=False)` — "an FBP reconstruction except it replaces the ramp filter with a derivative. For scans with angular ranges of 360 or more this will result in a pure noise reconstruction if the geometry is calibrated and there are no biases in the data. This can be used as a robust way to find the centerCol parameter or estimate detector tilt." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2845).
- Decomposed FBP: `filterProjections(g, g_out=None)` then `weightedBackproject(g, f)` reproduce `FBP` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2159, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2334). Finer decomposition: `preRampFiltering(g)`, `rampFilterProjections(g, get_FBPscalar())`, `postRampFiltering(g)`, `weightedBackproject(g, f)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2225, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2256).
- `HilbertFilterProjections(g)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2311), `rampFilterVolume(f)` (2D ramp filter per z-slice, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2367), `get_FBPscalar()` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2492).
- `space_carving(projection_mask, vol_mask)` — segmentation-reconstruction, `f_mask := 1 - u(P^T(1-g_mask))` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3572).

### 5.1 Filters and weightings

- `set_rampFilter(which)` — "Set the ramp filter to use: 0, 2, 4, 6, 8, 10, or 12... the order of the finite difference used in the ramp filter, higher numbers produce a sharper reconstruction. Shepp-Logan filter is the default value (2) and Ram-Lak is 12." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5831).
- `set_FBPlowpass(W)` — "Applies a low-pass filter of the specified FWHM to the ramp filter... must be >= 2.0" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5854).
- Parker weighting for short scans is documented as its own section of the manual (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex, section "Parker Weighting", label `sec:ParkerWeighting`); the implementation lives in https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/ray_weighting.cu / `ray_weighting_cpu.cpp`.
- Offset-scan and truncated-scan weighting via `set_offsetScan` / `set_truncatedScan` (section 2.7 above). `extraColumnsForOffsetScan()` reports the padding needed (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L413).
- Cone-beam artifact reduction: no dedicated algorithm, but a worked recipe using a Fourier data-fusion idea is provided (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d33_reducingConeBeamArtifacts.py, citing Leach et al., NDT & E International 127 (2022): 102600 and Stobbe et al., Sensing and Imaging 24, no. 1 (2023): 39).
- Row-direction extrapolation: "this script also demonstrates that the LEAP (non-helical) FBP reconstruction algorithms employ zeroth order extrapolation off the top and bottom of the detector" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d34_iterative_reconstruction_long_object.py); the `weightedBackproject` docstring says the same ("using extrapolation in the row direction for axial cone-beam FBP (FDK)").
- Scope claim: "LEAP contains an extensive collection of analytic inversion algorithms for parallel-, fan-, and cone-beam geometries with arbtirary detector shifts, offset center of rotation, non-equispaced projections angles, different angular ranges (short scan, full scan, over scan), axial or helical source trajectories, different voxels sizes and arbitary shifts of the volume location, etc." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex, section "Analytic Inversion Algorithms (FBP)").

---

## 6. Iterative reconstruction

### 6.1 Algorithm list

All are implemented in Python on top of the C++ projectors: "Python implementations of some iterative reconstruction algorithms: MLEM, OSEM, OS-SART, ASD-POCS, RWLS, RDLS, and ML-TR" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex, Introduction). Full list from https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/iterative_reconstruction.rst and https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py:

| Algorithm | Signature | Line | Notes |
| --- | --- | --- | --- |
| MLEM | `MLEM(g, f, numIter, filters=None, mask=None)` | 3606 | update `f_{n+1} = f_n/(P^T 1 + R'(f_n)) * P^T[g/(Pf_n)]`; "This reconstruction algorithms assumes the projection data, g, is Poisson distributed which is the correct model for SPECT data. CT projection data is not Poisson distributed because of the application of the -log" |
| OSEM | `OSEM(g, f, numIter, numSubsets=1, filters=None, mask=None)` | 3684 | ordered-subsets MLEM |
| SIRT | `SIRT(g, f, numIter, mask=None)` | 3773 | `f_{n+1} = f_n + (0.9/P^T 1) P^T[(Pf_n-g)/(P1)]`; "the same algorithm as a SART reconstruction with one subset" |
| SART | `SART(g, f, numIter, numSubsets=1, mask=None, nonnegativityConstraint=True)` | 3798 | ordered-subsets SIRT |
| ASD-POCS | `ASDPOCS(g, f, numIter, numSubsets, numTV, filters=None, mask=None, nonnegativityConstraint=True)` | 3904 | minimizes `R(f)` subject to `||Pf-g||^2 < eps`. "This function actually implements the iTV reconstruction method which is a slight varition to ASDPOCS which we find works slightly better." Reference given: Ritschl and Kachelriess, SPIE Medical Imaging 2011, vol. 7961, pp. 786-798 |
| LS | `LS(g, f, numIter, preconditioner=None, nonnegativityConstraint=True)` | 4092 | `0.5*||Pf-g||^2`, preconditioned conjugate gradient |
| WLS | `WLS(g, f, numIter, W=None, preconditioner=None, nonnegativityConstraint=True)` | 4119 | `0.5*(Pf-g)^T W (Pf-g)`; "if not given, W=exp(-g)" |
| RLS | `RLS(g, f, numIter, filters=None, preconditioner=None, nonnegativityConstraint=True)` | 4147 | LS + `R(f)` |
| RWLS | `RWLS(g, f, numIter, filters=None, W=None, preconditioner=None, nonnegativityConstraint=True)` | 4175 | WLS + `R(f)` |
| DLS | `DLS(g, f, numIter, preconditionerFWHM=1.0, nonnegativityConstraint=False, dimDeriv=2)` | 4391 | "same algorithm [as RDLS] without the regularization" |
| RDLS | `RDLS(g, f, numIter, filters=None, preconditionerFWHM=1.0, nonnegativityConstraint=False, dimDeriv=1)` | 4399 | `0.5*(Pf-g)^T Laplacian (Pf-g) + R(f)` |
| MLTR | `MLTR(g, f, numIter, numSubsets=1, filters=None, mask=None)` | 4561 | `<-t*log(exp(-Pf)) + exp(-Pf), 1>` with `t = exp(-g)`; "best models the noise for very low transmission/ low count rate data" |

The documentation groups them as Algebraic (SIRT, SART), Statistical-transmission (RWLS, MLTR), Statistical-emission (MLEM, OSEM), and Special-purpose for few-view/limited-angle (RDLS, ASDPOCS) (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/iterative_reconstruction.rst).

https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/LEAP_features.md item 11 additionally names "IFBP (RWLS-SARR)" — this is the `'SARR'` preconditioner used with RWLS, not a separate function.

### 6.2 Preconditioning

`LS`, `WLS`, `RLS`, `RWLS` take `preconditioner` as one of `'SQS'`, `'RAMP'`, `'SARR'` (docstrings, and demo https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d26_preconditioners.py):

- SQS: Separable Quadratic Surrogate, `(P* W P 1)^-1`. "This is a method made popular by Jeff Fessler... Note that the SART algorithm is a preconditioned gradient descent algorithm with constant step size, where W = 1/P1".
- RAMP: "uses a 2D ramp filter applied to each z-slice of a volume. This method was proposed by Clinthorne and Fessler and Booth."
- SARR: "Statistical-Analytic Regularized Reconstruction... proposed by myself (Kyle). This method convergences extremely fast, but only approximately minimizes the cost function."
- Caveat from the same demo: "The RAMP and SARR preconditioners should only be used when one has sufficient angular sampling and is not to be used for sparse-view CT reconstruction."
- `RDLS`/`DLS` use a different preconditioner: "The optional preconditioner is a 2D blurring for each z-slice", controlled by `preconditionerFWHM`.

### 6.3 Ordered subsets

- `numSubsets` argument on `SART`, `OSEM`, `MLTR`, `ASDPOCS`. Subset construction is in `subsetParameters` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7446) and `breakIntoSubsets(g, numSubsets)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3542).
- A commented-out warning in the source indicates subsets were once disallowed for modular-beam: `# print('WARNING: Subsets not yet implemented for modular-beam geometry, setting to 1.')` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3725, commented out).

### 6.4 Constraints, initialization, convergence control

- Non-negativity: `nonnegativityConstraint` boolean on `SART`, `ASDPOCS`, `LS`, `WLS`, `RLS`, `RWLS`, `DLS`, `RDLS`. Default is `True` for the LS family and `False` for DLS/RDLS.
- Volume masking (`set_volume_mask`) and projection-data masking (`mask` or `W` arguments) are the two masking mechanisms (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d35_volume_masking.py).
- Initialization: the caller supplies `f`, so any array can be used as a starting point. "if you first perform an FBP reconstruction and then an iterative reconstruction, like RWLS, then the iterative reconstruction will start with the FBP reconstruction; this trick can be used to accelerate an iterative reconstruction algorithm. If you want an iterative reconstruction to start from scratch, just initialize it with zeros" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d01_standard_geometries.py). RWLS additionally rescales a non-zero initial volume so that `<g,Pf>/<Pf,Pf>` matches (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L4222-L4231).
- Convergence is controlled only by `numIter`; there is no tolerance argument on any of the algorithms. RWLS uses a fixed conjugate-gradient restart: `conjGradRestart = 50` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L4211).
- Cost printing is a flag, not a stopping rule: `self.print_cost = False` by default, set True by `set_log_status()` / `set_log_debug()` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L174, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L220, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L229).
- Step sizes are computed with dedicated helpers `RWLSstepSize(...)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L4354) and `RDLSstepSize(...)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L4522), using the quadratic form of the regularizer (Separable Quadratic Surrogate).

### 6.5 Regularizers (filter sequence)

https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_filter_sequence.py, documented at https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/filter_sequence.rst. A `filterSequence(beta)` holds a weighted list; each filter has its own `weight`; "When used in a gradient-based algorithm, the cost of filter sequence is given by self.beta * sum_n filters[n].cost(f)".

Differentiable filters (implement `cost`, `gradient`, `quadForm`):

- `TV(leapct, delta=0.0, p=1.2, weight=1.0, f_0=None)` — anisotropic TV with a Huber-like loss `h(t)`; 6 or 26 neighbours via `set_numTVneighbors(N)`. "if specified this class calculates TV(f-f_0), e.g., PICCS". Guidance on `delta`: "It is recommended that one set the value of delta to be smaller than the difference of two materials that one wishes to discriminate between... I usually start with dividing this difference by 20" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_filter_sequence.py#L193).
- `LpNorm(leapct, delta=0.0, p=1.0, weight=1.0, f_0=None, FWHM=0.0)` — Huber-like loss on `B(f-f_0)` where B is identity, a low-pass (FWHM > 1.0) or a high-pass (FWHM < -1.0) filter (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_filter_sequence.py#L270).
- `histogramSparsity(leapct, mus=None, weight=1.0)` — "encourages sparisty in the histogram domain. Warning: this is a nonconvex regularizer. It is best to use this filter after an initial reconstruction is performed." Uses Geman functions with a parameter derived from the minimum spacing of the target values (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_filter_sequence.py#L431).
- `azimuthalFilter(leapct, FWHM, p, weight=1.0)` — "applies a high pass filter in the azimuthal direction of the reconstructed volume. If is useful for reconstructing objects that have a sparse azimuthal gradient, such as pipes with delaminations, voids, or high density inclusions." Functional is `||H(f)||_p^p` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_filter_sequence.py#L511).

Non-differentiable filters (implement `apply` only; "those filters that are not differentiable will be ignored by gradient-based iterative reconstruction algorithms. Currently the only non-gradient-based iterative reconstruction algorithm in LEAP is ASDPOCS" — https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/filter_sequence.rst):

- `BlurFilter(leapct, FWHM)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_filter_sequence.py#L96), `BilateralFilter(leapct, spatialFWHM, intensityFWHM, scale=1.0)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_filter_sequence.py#L119), `GuidedFilter(leapct, r, epsilon)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_filter_sequence.py#L148), `MedianFilter(leapct, threshold=0.0, windowSize=3)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_filter_sequence.py#L174), `SparseDictionary(leapct, dictionary, sparsityThreshold=8, epsilon=0.0, f_0=None)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_filter_sequence.py#L576).

Standalone denoisers exposed on `tomographicModels` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/filters.rst), all working on any 3D array:

- `LowPassFilter`, `LowPassFilter2D`, `HighPassFilter`, `HighPassFilter2D`, `BlurFilter`, `BlurFilter2D`, `MedianFilter`, `MedianFilter2D`, `MeanFilter`, `VarianceFilter`, `LowSignalCorrection`, `LowSignalCorrection2D`, `BilateralFilter`, `PriorBilateralFilter`, `GuidedFilter`, `DictionaryDenoising`, `diffuse`, `TV_denoise`, `TVcost`, `TVgradient`, `TVquadForm`, `AzimuthalBlur` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L4889 onward).
- Anisotropic TV definition, from the manual: `R_delta(f) = sum_i sum_{j in N(i)} ||i-j|| h_delta(f_i - f_j)`, with `h_delta(t) = t^2/2` for `|t| <= delta` and `(delta^2/6)[5|t/delta|^1.2 - 2]` otherwise (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex, section "Anisotropic Total Variation"). Note the manual states `||i-j||` while the Python docstring states `||i-j||^{-1}` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5385); the two write the weight differently.
- LEAP_features item 12: "Fast multi-GPU 3D densoing methods." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/LEAP_features.md).

---

## 7. Preprocessing and physics

### 7.1 In-repo preprocessing (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py, listed in https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/preprocessing_algorithms.rst)

- `gain_correction(leapct, g, air_scan, dark_scan, calibration_scans, ROI, badPixelMap, flux_response)` — "processes raw radiographs by subtracting off the dark current and correcting for the pixel-to-pixel gain variations which reduces ring artifacts" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L26). Limitation in the code: "Error: current implementation only works for 3 calibration scans!" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L150).
- `makeAttenuationRadiographs(leapct, g, air_scan, dark_scan, ROI, isAttenuationData)` — flat fielding and negative log: `transmission = (raw - dark)/(air - dark)`, `attenuation = -log(transmission)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L176).
- `badPixelCorrection(leapct, g, air_scan, dark_scan, badPixelMap, windowSize, isAttenuationData)` — median filter at flagged pixels; "If no bad pixel map is provided, this routine will estimate it from the average of all projections." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L287).
- `outlierCorrection(leapct, g, threshold, windowSize, isAttenuationData)` — thresholded median filter for zingers (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L341).
- `outlierCorrection_highEnergy(leapct, g, isAttenuationData)` — "a series of three thresholded median filters... should most be used for MV CT or neutron CT where outliers effect a larger neighborhood of pixels" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L373).
- `LowSignalCorrection(...)` — photon-starvation correction by double-thresholded median filter (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L406).
- `detectorDeblur_FourierDeconv(leapct, g, H, WienerParam, isAttenuationData)` — Wiener deconvolution (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L439).
- `detectorDeblur_RichardsonLucy(leapct, g, H, numIter, isAttenuationData)` — "developed for Poisson-distributed data and inherently preserve the non-negativity of the input" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L466).
- Ring removal, three variants: `ringRemoval_fast` (TV-denoise the mean projection; "runs fast, but can sometimes create new rings of tangents of sharp transitions") (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L506), `ringRemoval_median` (median-filter based; "effective at removing ring artifacts without creating new ring artifacts") (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L578), and `ringRemoval` (TV gradient averaged over angles; "more computationally expensive than ringRemoval_fast") (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L619).
- `transmission_shift(leapct, g, shift, isAttenuationData)` — "Subtracts constant from transmission data which is a simple method for scatter correction" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L686).
- `expNeg(x)` / `negLog(x, gray_value=1.0)` on `tomographicModels` for transmission/attenuation conversion (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3489, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3514).

### 7.2 Metal artifact reduction

- `sinogram_replacement(g, priorSinogram, metalTrace, windowSize=None)` — "This routine provides a robust solution to metal artifact reduction (MAR)." Default window `[30, 1, 50]` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1343).
- Demo https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d30_MAR_and_sinogram_replacement.py credits "sinogram replacement... a method developed by Seemeen Karimi" and also shows RWLS with metal-trace-based weights.

### 7.3 Scatter

- `scatter_model(f, source, energies, detector, sigma, scatterDist, jobType=-1)` — "simulates first order scatter through an object composed of a single material type (but variable density)" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1473). Stated size limits: "The volume should be no larger than 200^3 voxels. The projection data should be no larger than 256^2. The source spectra should have no more than 20 samples (we recommend about 10 samples)".
- "The one in LEAP is physics-based first order scatter. This means that the scatter signal is calculating using physics models, rather than heuristic-based kernel methods. This type of scatter correction is highly accurate, but computationally expensive." The workflow down-samples, estimates, and up-samples (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d31_scatter_correction.py).

### 7.4 Beam hardening, dual energy, spectra (requires XrayPhysics)

- "The following physics-based algorithms requires the XrayPhysics software package" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/physics_based_preprocessing_algorithms.rst), listing single-material scatter correction, single-material BHC, two-material BHC, and dual-energy decomposition / SIRZ.
- `applyTransferFunction(x, LUT, sampleRate, firstSample)` — 1D lookup transfer function applied to 2D or 3D data (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L4736).
- `beam_hardening_heel_effect(g, anode_normal, LUT, takeOffAngles, sampleRate, firstSample)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L4765); "Error: beam_hardening_heel_effect not yet implemented for torch tensors!" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L4781).
- `applyDualTransferFunction(x, y, LUT, sampleRate, firstSample)` — 2D lookup applied to a data pair (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L4789).
- `convertToRhoeZe(f_L, f_H, sigma_L, sigma_H)` — "transforms a low and high energy pair to electron density and effective atomic number" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L4829). Not implemented for torch tensors ("Error: not yet implemented for pytorch tensors", https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L4878).
- Demos: `d17_beam_hardening.py` (single-material BHC), `d18_multi-materialBHC.py`, `d19_dual_energy_decomposition_and_SIRZ.py`, `d37_spectral_calibration.py` (spectral calibration from reference materials; "We will make changes to this demo script in the future to make it more robust and run faster.").
- README states as future work: "multi-material beam hardening correction algorithms for more than two materials and that account for variable takeoff angle and graded collimator/ bowtie filter", "triple energy decomposition", "spectral calibration" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/README.md, "Future Releases").

### 7.5 Geometric calibration

- `find_centerCol(g, iRow=-1, searchBounds=None)` — minimizes the difference of conjugate rays; explicit cost functions given for parallel and fan beam. "Note that this only works for parallel-, fan-, and cone-beam CT geometry types (i.e., everything but modular-beam) and one may not get an accurate estimate if the projections are truncated" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L846).
- `find_tau(g, iRow=-1, searchBounds=None)` — "only works for fan- and cone-beam CT geometry types" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L892).
- `estimate_tilt(g)` — detector rotation about the optical axis, via conjugate-projection differences after rebinning to parallel/cone-parallel (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L937).
- `conjugate_difference(g, alpha=0.0, centerCol=None)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L976).
- `consistency_cost(g, Delta_centerRow, Delta_centerCol, Delta_tau, Delta_tilt)` — "only works for the axial flat-panel cone-beam CT geometry type", implementing Lesaint, Rit, Clackdoyle and Desbat, IEEE TRPMS 1, no. 6 (2017): 517-526 (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1021). C++ guard: "Consistency metric only works for axial flat panel cone-beam geometries!" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/geometric_calibration.cu#L192).
- `geometric_calibration(leapct, g, shifts, tilts, param, method, iz)` — joint search over centerCol-or-tau and tiltAngle, with metric `'inconsistency'` or `'bowtie'` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L737).
- `parameter_sweep(leapct, g, values, param, iz, algorithmName, set_optimal, isFiltered)` — sweeps `'centerCol'`, `'centerRow'`, `'tau'`, `'sod'`, `'sdd'`, `'tilt'`, `'vertical_shift'`, `'horizontal_shift'` and returns a stack of single-slice reconstructions (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L872).
- `ball_phantom_calibration` class — least-squares fit against a ball phantom with known spacing (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L1159; demo `d36_ball_phantom_calibration.py`).
- `MTF(leapct, f, r, center, getEdgeResponse, oversamplingRate)` — resolution measurement utility (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L1087).
- These calibration routines are CPU-only for the data: "Error: find_centerCol not yet implemented for data on the GPU", "find_tau not yet implemented for data on the GPU", "estimate_tilt not yet implemented for data on the GPU", "conjugate_difference not yet implemented for data on the GPU" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L3932, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L3943, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L3954, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L3965).

---

## 8. Simulation

- `addObject(f, typeOfObject, c, r, val, A=None, clip=None, oversampling=1)` — geometric primitives, "ELLIPSOID=0, PARALLELEPIPED=1, CYLINDER_X=2, CYLINDER_Y=3, CYLINDER_Z=4, CONE_X=5, CONE_Y=6, CONE_Z=7", with an optional 3x3 rotation matrix and clipping planes. Objects are either voxelized into `f` or pushed onto a stack for analytic ray tracing. "Background objects must be specified first and foreground objects defined last." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7304).
- `rayTrace(g=None, oversampling=1)` — "Performs analytic ray-tracing simulation through a phantom composed of geometrical objects" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7261).
- `voxelize(f, oversampling=1)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7286), `clearPhantom()` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7417), `scalePhantom(c)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7422).
- `set_FORBILD(f=None, includeEar=True, oversampling=1)` — FORBILD head phantom. "Note that the values of the FORBILD head phantom are all scaled by 0.02 which is the LAC of water at around 60 keV. The FOV is about [-96, 96, -120, 120, -125, 125]" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7357).
- Ray oversampling models partial volume: "one may employ ray oversampling in the analytic ray tracing methods to model the non-linear partial volume effect... `g = -log( (sum_{n=1}^N exp(-rayTrace sub ray n)) / N )` i.e., the averaging is done in transmission space which is a more accurate way to model real CT data" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d08_ray_tracing_simulation.py).
- The same demo makes the inverse-crime point: "Simulating CT data by forward projecting a voxelized phantom is known as an 'inverse crime'... performing simulations using analytic ray tracing methods results in a better assessment of various CT reconstruction methods".
- Noise: no built-in noise model. The demos add Poisson noise in numpy, e.g. `I_0 = 50000.0; g[:] = -np.log(np.random.poisson(I_0*np.exp(-g))/I_0)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d01_standard_geometries.py).
- Polychromatic simulation, source spectra, detector response and attenuation cross-sections come from the external XrayPhysics package, e.g. `physics.simulateSpectra(100.0, 11.0)`, `physics.detectorResponse('GOS', None, 0.1, Es)`, `physics.filterResponse('Al', None, 2.0, Es)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d37_spectral_calibration.py).
- Detector blur is modelled through `transmission_filter(g, H, isAttenuationData)` and the deblur routines, with `H` a user-supplied 2D frequency response (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2444; demo `d15_detector_deblur.py`). No focal-spot blur model was found.
- `VAM/VAM.py` — a 706-line Volumetric Additive Manufacturing solver built on LEAP's Attenuated Radon Transform, "Algorithm based on: Numerical Optimization of the Light Intensity Fields used in Volumetric Additive Manufacturing by Kyle Champley, Erika Fong, and Maxim Shusteff" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/VAM/VAM.py#L1-L20).

---

## 9. Deep-learning integration (leaptorch)

Source: https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leaptorch.py (512 lines). Documented at https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/leaptorch.rst (an `automodule` directive only).

- Tensor layouts, stated as comments at the top of the file: "Image tensor format: [Batch, ImageZ, ImageY, ImageX]" and "Projection tensor format: [Batch, Views, Detector_Row, Detector_Col]" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leaptorch.py#L15-L17).
- Six `torch.autograd.Function` subclasses: `ProjectorFunctionCPU`, `ProjectorFunctionGPU`, `BackProjectorFunctionCPU`, `BackProjectorFunctionGPU`, `FBPFunctionCPU`, `FBPFunctionGPU`, plus `FBPReverseFunctionCPU` / `FBPReverseFunctionGPU`.
- Gradients are supplied by the adjoint operator, not by autograd tracing: the backward of forward projection calls `lct.backproject_gpu(g, f, param_id)` and vice versa (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leaptorch.py#L31-L36, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leaptorch.py#L50-L55).
- FBP's backward uses `fbp_adjoint`, with an in-source caveat: `lct.fbp_adjoint_gpu(g, f) # compute proj (g) from input (f) -> needs to be replaced!!!` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leaptorch.py#L139; same comment on the CPU path at https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leaptorch.py#L120).
- Batching is a Python loop, one projector call per batch element: `for batch in range(input.shape[0]): ... lct.project_gpu(g, f, param_id.item())` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leaptorch.py#L41-L45).
- Output buffers are preallocated module state, not freshly allocated per call: `self.proj_data` and `self.vol_data` are created by `set_volume` / `set_default_volume` / the geometry setters / `allocate_batch_data()`, and the autograd functions write into them and return them (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leaptorch.py#L186, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leaptorch.py#L210-L230).
- `BaseProjector(torch.nn.Module)` exposes `set_volume`, `set_default_volume`, `set_parallelbeam`, `set_fanbeam`, `set_conebeam` (with `helicalPitch`), `set_modularbeam`, `get_volume_dim`, `get_projection_dim`, `allocate_batch_data`, `load_param`, `save_param`, `set_gpu`, `set_gpus`, `print_parameters`. Note `set_coneparallel` is absent from the torch wrapper.
- `Projector(forward_project=True, use_static=False, use_gpu=False, gpu_device=None, batch_size=1)` — `forward()` does forward projection when `forward_project=True` and backprojection otherwise; `fbp(input)` performs FBP without autograd (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leaptorch.py#L430).
- `FBP(forward_FBP=True, ...)` — a module whose forward is FBP (backward = FBP adjoint) or the reverse.
- The `tomographicModels` object is reachable as `proj.leapct`, so the whole non-torch API is available from inside a network (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/INTRO.md).
- Training example given in the README-level docs:
  ```
  optimizer = Adagrad(self.nn.parameters(), lr=float(self.learning_rate))
  for i in range(N_iter):
      optimizer.zero_grad()
      img_pred = network(img_init)
      sino_pred = proj(img_pred)
      loss = loss_func(sino_pred.float(), sino_gt.float())
      loss.backward()
      optimizer.step()
  ```
  (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/INTRO.md).
- Demos in https://github.com/LLNL/LEAP/tree/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leaptorch: `test_project_and_FBP.py` (158 lines), `test_recon_NN.py` (310 lines, network modes "0: no network used, 1: fully connected only, 2: convolutional with fully connected"), `test_recon_TV.py` (265 lines, FISTA + TV, citing Beck and Teboulle, IEEE TIP 18(11):2419-2434, 2009), `test_recon_projections_NN.py` (369 lines, solving for projections from a volume via FBP or backprojection), and helper `TVGPUClass.py` (104 lines).
- A separate https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leaptorch_xyz.py (278 lines) is an XYZ-volume-order variant that imports a compiled `leapct` extension module rather than the ctypes library; it is not referenced by `setup.py`'s `py_modules` list.
- The non-torch API also accepts torch tensors directly, including tensors already on a GPU: "This API works with both numpy and torch tensors. The torch tensors allow the LEAP algorithms to operate directly on data that is already on a GPU and thus does not rely on CPU-to-GPU data transfers." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/INTRO.md). Demo `d02_standard_geometries_torch.py` runs iterative reconstructions on GPU tensors.
- Constraint when data is already on a GPU: "However they are limited to the amount of GPU memory one has and only process on one GPU at a time." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d02_standard_geometries_torch.py).

---

## 10. Compute: CUDA, multi-GPU, CPU, memory

### 10.1 Build and dependency facts

- Dependencies (from the wiki, https://github.com/LLNL/LEAP/wiki/Installing-and-Using-LEAP): PyTorch; "cmake version 3.23.3 or newer"; "CUDA toolkit 11.7 or newer"; gcc on Linux; Visual Studio 2019 on Windows.
- https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/CMakeLists.txt requires `cmake_minimum_required(VERSION 3.18)`, `find_package(CUDA 11.7 REQUIRED)`, `find_package(OpenMP REQUIRED)`, C++17.
- CUDA architectures: `set_property(TARGET leapct PROPERTY CUDA_ARCHITECTURES all-major)` when CMake > 3.23 (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/CMakeLists.txt).
- Links against `${CUDA_LIBRARIES}`, `${CUDA_cublas_LIBRARY}`, `${CUDA_cufft_LIBRARY}` and `OpenMP::OpenMP_CXX`.
- Python dependencies declared: `install_requires=['numpy', 'torch']` in https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/setup.py; `install_requires=['numpy']` in https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/setup_ctype.py (the PyTorch-free build). `python_requires='>=3.6'`.
- https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/LEAP_features.md item 16: "Easy-to-build executable because the only dependency is CUDA. Python API can be run with or without PyTorch (of course the neural network stuff requires PyTorch)."
- The Python file also imports `imageio` unconditionally (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L16) and uses `napari` for `display` and `matplotlib` for `sketch_system`.
- Docker: `FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel`, then `git clone` and `pip install .` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/Dockerfile).

### 10.2 CPU support

- `set_gpu(-1)` switches to CPU: "Set which GPU to use, use -1 to do CPU calculations" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5627).
- CPU builds use `-D__USE_CPU` and a separate https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/cpu_CMakeLists.txt with no CUDA. The wiki instructs: "If you are building on a Mac, you will have to install gcc. First, you need to swap the CMake file by renaming cpu_CMakeLists.txt to CMakeLists.txt (in the src folder)." (https://github.com/LLNL/LEAP/wiki/Installing-and-Using-LEAP).
- CPU is OpenMP multi-threaded ("multi-core CPU implementations of all algorithms", https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/LEAP_features.md item 3).
- CPU feature gaps found in the source:
  - Cone/fan/parallel CPU projection requires the SF voxel-size condition, otherwise it refuses: "Error: The voxel size for CPU-based cone-beam projectors must be closer to the nominal size. Please either change the voxel size or use the GPU-based projectors." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors.cpp#L92-L99, and equivalent messages for fan and parallel).
  - "Error: CPU-based projector not yet implemented for this geometry." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors.cpp#L129) and the matching backprojector message at https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors.cpp#L229 — this covers cone-parallel on CPU.
  - "Error: attenuated radon transform for voxelize attenuation map only works for GPU projectors!" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors.cpp#L182).
  - "Error: CPU-based FBP not yet implemented for helical cone-beam geometry" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/filtered_backprojection.cpp#L441).
  - Many post-processing and denoising routines are GPU-only. The string "Error: this function is currently only implemented for GPU processing!" appears at https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L2690, 2756, 2822, 2901, 2982, 3057, 3123, 3202, 3412, 3490, 3570, 3648, 3705, 4013, 4076 — 15 sites.
- No performance figure for the CPU path is stated anywhere in the repository.

### 10.3 Apple Silicon / macOS

- README footnote: "*Mac version does not have GPU support and some featurings are missing." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/README.md). The introductory line says the library is for "(Linux, Windows, and Mac*)".
- https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/setup.py handles `_platform == "darwin"` by building `build/lib/libleapct.dylib` via `etc/build.sh`; https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L141 looks for `*leapct*.dylib` on Darwin.
- No macOS binary is published: release assets for every release are only `libleapct.dll` and `libleapct.so` (GitHub API `repos/LLNL/LEAP/releases`). There is no `.dylib` asset in any of the 27 releases.
- Nothing in the repository mentions Apple Silicon, arm64, Metal, or MPS specifically. macOS support means the CPU build only.

### 10.4 AMD GPUs

- https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/setup_AMD.py exists and detects ROCm at build time: `rocm = "AMD" in torch.cuda.get_device_name(0)`, though the compile flags for the ROCm and CUDA branches are currently identical.
- https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/hip_utils.h, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors_Joseph_cpu_hip.h, and https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/ramp_filter_hip.cuh are present in the tree.
- A dedicated `AMD` branch exists, last commit 2025-05-16, "update kernensl for AMD (no texture memory GPU)" (GitHub API `repos/LLNL/LEAP/commits?sha=AMD`).
- AMD support is explicitly listed as *future* work on main: "For the next releases, we are working on the following: ... 2) AMD GPU Support" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/README.md).

### 10.5 Multi-GPU

- `set_gpus(listOfGPUs)` / `set_gpu(which)` / `set_all_gpus()` / `number_of_gpus()` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5641, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5627, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5638, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5623).
- Default behaviour: "The default setting is for LEAP to use all GPUs, so if this is what you want there is no need to run the set_gpus function" (`tomographicModels.__init__` docstring, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L39).
- Parallelism is OpenMP over chunks, one chunk per GPU: `omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs())); #pragma omp parallel for schedule(dynamic)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L1003-L1005, and equivalents in the other multi-GPU functions).
- Splitting strategy:
  - Forward projection: split over **detector rows** (`project_multiGPU`, `numRowsPerChunk`, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L734-L800).
  - Backprojection and FBP: split over **volume z-slices** (`backproject_FBP_multiGPU`, `numSlicesPerChunk`, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L1090-L1160).
  - Split over **views** instead when the geometry cannot be sliced that way: helical cone-beam or cone-parallel, and non-axially-aligned modular beam (`project_multiGPU`, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L740-L747; `project_multiGPU_splitViews` at https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L962).
- Preconditions for multi-GPU: `if (params.volumeDimensionOrder != parameters::ZYX || params.isSymmetric()) return false;` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L737). So multi-GPU is disabled for XYZ volume order and for symmetric-object reconstructions.
- Multi-GPU only applies when the data starts on the CPU: "Providing data that is on the CPU is best for medium to large data sets because this allows for multi-GPU processing and LEAP will automatically divide up the data into small enough chunks so that is does not exceed the GPU memory." vs "If the data is on the GPU (only possible with torch tensors): then the computations must also take place on this particular GPU" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L39 docstring).
- Small jobs are not split: chunking is skipped when one GPU is listed or when `requiredGPUmemory(...) <= params.chunkingMemorySizeThreshold`, where `chunkingMemorySizeThreshold = float(0.1)` GB (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/parameters.cpp#L60; used at https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L784, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L992, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L1139).

### 10.6 Memory management and out-of-core

- Claim: "Algorithms not limited by the amount of GPU memory." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/LEAP_features.md, item 4).
- Chunk-size controls: `maxSlicesForChunking = 128` and `minSlicesForChunking = std::min(32, maxSlicesForChunking)` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L46, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L51).
- User control: `set_maxSlicesForChunking(N)` — "Smaller numbers use less GPU memory, but may slow down processing. Only use this function if you know what you are doing. For forward projection it specifies the maximum number of detector rows used per job. For backprojection it specifies the maximum number of CT volume z-slices used per job." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L254).
- Automatic halving loop when memory is short: `while (memAvailable < memNeeded) { numRowsPerChunk = numRowsPerChunk / 2; ... }` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L779-L787 for projection, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L1108-L1121 for backprojection).
- Pre-flight check with an explicit error: "Error: insufficient GPU memory" followed by available and required memory (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors.cpp#L66-L70, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors.cpp#L148-L158).
- Chunking is over GPU memory, not host memory. For host-memory limits the recommended approach is manual cropping: `crop_cols`, `crop_rows`, `crop_projections`, which update the geometry parameters automatically (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d13_cropping_subchunking.py). LEAP itself has no memory-mapped or streaming-from-disk mode. Disk-backed chunking exists only in the separate GUI project (see section 22): LEAPCT-UI-GUI "manages memory by using hard drive storage when CPU RAM is insufficient".
- Known open issue about GPU memory: "Increasing GPU memory usage caused by the iterative use of FBP" (issue #125, opened 2024-10-27) and "cuda memory leak" (issue #170, opened 2025-05-24). A https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/unitTests/memory_leak_tests.py script exists (50 lines).

---

## 11. Data I/O

- Supported formats, from `save_projections` / `save_volume` / `load_data`: "tif sequence, nrrd, or npy" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L6709, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L6742, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7092).
- `load_data(fileName, x=None, fileRange=None, rowRange=None, colRange=None, axis_split=0)` — "A tif sequences must be in the following form: basename_XXXX.tif or (tiff). The XXXX are the sequence numbers which can be padded with zeros or not... Note that fileRange, rowRange, and colRange arguments only apply to tif sequences." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7092).
- `load_projections`, `load_volume`, `save_projections`, `save_volume`, `save_data`, `load_tif`, `save_tif`, `get_file_list` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7036, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7025, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L6709, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L6742, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L6926, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L6808, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L6894, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7047).
- `set_fileIO_parameters(dtype=np.float32, wmin=0.0, wmax=None)` — "the data type to use, can be: np.float32, np.uint8, or np.uint16" with clip window; "If dtype is np.float32, the data is not clipped" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L232).
- Parameters (metadata) are saved and loaded as a text file: `save_parameters(fileName)` / `load_parameters(parameters_fileName, param_type=0)` where "param_type (int): if 0, assumes that parameters_fileName is a file; if 1, assumes that parameters_fileName is the actual content of a parameters file" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L6695, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L6564). A sample file is https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/sample_data/conebeam_geometry.txt.
- Recommendation from the demo: "We also recommend using the nrrd format which is a file format for N-dimension data which is readable by many common 3D image viewing software such as ImageJ and 3D slicer. Of course you can also use npy files, but these are only good for python and are not always supported by 3D viewers" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d11_fileIO.py).
- **No DICOM support** was found. Greps of `src/*.py` and https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/file_io.cpp for "dicom"/"DICOM" returned nothing, and no vendor-specific reader (Varian, Siemens, GE, Zeiss, Nikon, etc.) is present.
- Geometry bridges to other packages, in https://github.com/LLNL/LEAP/tree/0c8846f42b2e59340d5559fc1271d590a292f9a0/utils:
  - `tigre_geometry_bridge.py` (147 lines) — `set_leap_from_tigre(geo, leapct=None)` and the reverse. Header warning: "I do not gaurantee that this works correctly in all cases and some features may be missing!!!".
  - `bridgeToLTT.py` (193 lines) — conversion to and from LLNL's closed-source LTT. "Some parameters mappings are not yet implemented, such as non equi-spaced projection angles."
  - `generateDictionary.py` (194 lines), `symmetric_projectors.py` (1022 lines).

---

## 12. Visualization and GUI

- `display(vol)` / `display_volume(vol)` / `displayVolume(vol)` — "Uses napari to display the provided 3D data" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L6255).
- `sketch_system(whichView=None)` — "Uses matplot lib to sketch the CT geometry and CT volume" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L6279). Manual figure caption: "the magenta box is the reconstruction volume, the gray rectangle is the detector panel, the green dots are the source positions, and the red lines trace from the source to each of the four corners of the detector" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex).
- `print_parameters()` prints the full geometry and volume specification (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L327).
- PyQt GUI is a separate repository: https://github.com/kylechampley/LEAPCT-UI-GUI (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/README.md, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/LEAP_features.md item 15).

---

## 13. API ergonomics

### 13.1 Minimal example

From https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/INTRO.md:

```python
from leapctype import *
leapct = tomographicModels()
```

A complete minimal run, condensed from https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d01_standard_geometries.py:

```python
from leapctype import *
leapct = tomographicModels()

numCols = 512
numAngles = 2*2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols
numRows = numCols

leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize,
                    0.5*(numRows-1), 0.5*(numCols-1),
                    leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
leapct.set_default_volume()

g = leapct.allocate_projections()   # shape is numAngles, numRows, numCols
f = leapct.allocate_volume()        # shape is numZ, numY, numX
leapct.set_FORBILD(f, True)

leapct.project(g, f)
f[:] = 0.0
leapct.FBP(g, f)
leapct.display(f)
```

And the PyTorch entry point, from https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/INTRO.md:

```python
from leaptorch import Projector
proj = Projector(forward_project=True, use_static=True, use_gpu=use_cuda, gpu_device=device)
```

### 13.2 Design notes

- LEAP owns no data. "All memory for data structures, e.g., the projection data and the volume data are managed in python. LEAP only tracks the specifications, i.e., geometry of the CT model, the volume parameters, and a few other parameters that deal with how the code should be run" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d01_standard_geometries.py). The manual says the same: "The only permanent memory that LEAP manages itself is a class with member variables that parameterize the CT geometry and CT volume." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex).
- Parameter names are physical and flat: `numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau, helicalPitch, tiltAngle`; volume: `numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ`. Every parameter has a matching `get_*` accessor (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L6006-L6234).
- Multiple concurrent models: each `tomographicModels()` instance creates a new parameter set in the C++ layer; `param_id` can be shared between objects. Demo `d24_circle_plus_line.py` uses two instances at once.
- Extensive aliasing for backward compatibility: `set_coneBeam`/`set_conebeam`, `set_GPU`/`set_gpu`, `allocateVolume`/`allocate_volume`, `printParameters`/`print_parameters`, etc.
- Logging levels: `set_log_error`, `set_log_warning`, `set_log_status`, `set_log_debug` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L194-L232).
- `reset()`, `copy_parameters(leapct)`, `about()`, `version()` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L286, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L274, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L298, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L303).

### 13.3 Documentation, demos, tests, CI

- Sphinx documentation source: 18 `.rst` files under https://github.com/LLNL/LEAP/tree/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source, mostly `autofunction` directives against the docstrings. Built by readthedocs per https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/.readthedocs.yaml (Ubuntu 22.04, Python 3.12).
- Technical manual: https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.pdf with LaTeX source https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex (717 lines). Its title page says "LEAP Technical Manual, Version 1.1" while the code is at 1.26, so the manual version and the code version are not synchronized.
- Demo scripts: 38 Python files in https://github.com/LLNL/LEAP/tree/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype (`d01`-`d37`, plus `d98_SF_vs_VD.py` and `d99_speedTest.py`; there is no `d20`). 5 files in https://github.com/LLNL/LEAP/tree/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leaptorch. Most demos carry a multi-paragraph explanatory docstring.
- Tests: https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/unitTests/unit_tests.py (206 lines), https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/unitTests/verificationTests.py (136 lines), https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/unitTests/memory_leak_tests.py (50 lines). These are scripts, not a pytest suite; `unit_tests.py` has `geometries = []` assigned right before the main loop (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/unitTests/unit_tests.py#L40), which disables the loop as checked in. `verificationTests.py` requires the closed-source LTT package and contains hard-coded Windows paths (`C:\Users\champley\Documents\git_leap\LEAP\utils`).
- **No CI**: there is no `.github` directory in the repository (verified by `ls -a`). No GitHub Actions, no test workflow, no build matrix.
- `ENABLE_TESTING()` is called in the top-level https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/CMakeLists.txt but no `add_test` targets exist.

---

## 14. Performance claims found in the repository

Every claim below is quoted exactly. **None of them names hardware, and none names a problem size for the timing claims.**

1. "There are a lot of CT reconstruction packages out there, so why choose LEAP? In short, LEAP has more accurate projectors and FBP algorithms, more features, and most algorithms run as fast or faster than other popular CT reconstruction packages" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/README.md).
2. "As a simple demonstration of the accuracy of our projectors we show below the results of FDK reconstructions using ASTRA and LEAP of the walnut CT data. **The LEAP reconstruction has 1.7 times higher SNR than ASTRA.**" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/README.md). The script that produced it is https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d21_walnut.py; the dataset is the public walnut CT data and "We chose the reconstruction volume size to match what they did in the ASTRA script for comparison purposes". No hardware, timing, or SNR definition is given.
3. "**Multi-GPU and multi-core CPU implementations of all algorithms** that are as fast or faster than other popular CT reconstruction packages." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/LEAP_features.md, item 3).
4. "Briefly, SF projectors are more accurate, but VD backprojection is faster. For example, **VD backprojection of cone-beam data is about twice as fast as SF-based backprojection in LEAP.**" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/results/SF_vs_VD.md).
5. "Indeed, just as predicted, the VD result has high resolution, but the SF result has higher SNR. **The SNR for the VD result is 25.7 and the SNR for the SF result is 43.6.**" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/results/SF_vs_VD.md, in the "Voxels that are larger than the nominal size" section, i.e. voxel size = `2.0 * sod/sdd * pixelWidth`, with noise added to the projection data). The script is https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d98_SF_vs_VD.py (180 lines).
6. "Our models are slightly faster than the original rectangle-rectangle SF method because they utilize the bilinear interpolation capabilities of texture memory, but are very, very slightly less accurate." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex, section "Matched Projector Models").
7. "If the dictionary is complete and orthonormal, the algorithm runs **about 2-8 times faster**." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d27_dictionaryDenoising.py, about `DictionaryDenoising`).
8. "One difference in LEAP is that our cone-beam geometry is much, much more flexible that ASTRA's cone geometry. Thus, most cone-beam geometries should be covered by LEAP's standard cone-beam geometry which is much more simple to specify and **the algorithms (e.g., forward and back projectors) are faster**" than the modular-beam ones. Also: "The modular-beam projectors are not as fast and not as accurate" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d04_modularbeam.py, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d05_3DflashCT.py).
9. Warning against off-nominal voxel sizes: "Using voxel sizes that are significantly smaller or significantly bigger than this default size may result in poor computational performance." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d01_standard_geometries.py), and `set_default_volume(scale)` is annotated "(not recommended for fast reconstruction)".
10. Benchmark harness present but with no recorded results: https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d99_speedTest.py — "The only purpose of this demo script is to evaluate the speed of LEAP's forward projections, backprojection, and FBP algorithms." Its default configuration is `N = 1024`, `M = N`, `numAngles = int(720*N/1024)` (= 720), `pixelSize = 0.2*2048/N`, cone-beam with `sod=1100`, `sdd=1400`, `tiltAngle=0.2`, `set_projector('VD')`, and it prints projection, backprojection, FBP, and filter-projections times. The repository contains no output from it.

The https://github.com/LLNL/LEAP/tree/0c8846f42b2e59340d5559fc1271d590a292f9a0/results directory contains the supporting images `MTF_0.png`, `MTF_1.png`, `MTF_2.png`, `noisey_0.png`, `noisey_1.png`, `noisey_2.png`, and `walnut_comparison.png`, but no numeric log files.

---

## 15. Project health

- Stars: 249; forks: 36; watchers: 249; subscribers: 10; network count 36 (GitHub API `repos/LLNL/LEAP`, retrieved 2026-09-02).
- Issues: 196 total, 44 open, 152 closed (GitHub search API `repo:LLNL/LEAP type:issue`, retrieved 2026-09-02). The repository's `open_issues_count` field reads 47, which includes open pull requests.
- Discussions: 13 total (GitHub GraphQL `repository.discussions.totalCount`).
- Contributors: `kylechampley` (430 commits), `hkimdavis` (7), `kchampley` (1) (GitHub API `repos/LLNL/LEAP/contributors`). Effectively a single-maintainer project.
- Releases: 27 releases, v1.0 through v1.26. Latest is **v1.26, published 2024-12-14T23:24:36Z**.
- The entire release history falls inside calendar 2024: v1.0 on 2024-01-07 and v1.26 on 2024-12-14. Cadence was roughly every two to four weeks (v1.1 2024-01-21, v1.2 2024-01-27, v1.3 2024-02-04, v1.4 2024-02-12, v1.5 2024-03-02, v1.6 2024-03-08, v1.7 2024-03-16, v1.8 2024-04-11, v1.9 2024-04-30, v1.10 2024-05-05, v1.11 2024-05-18, v1.12 2024-05-25, v1.13 2024-06-04, v1.14 2024-06-09, v1.15 2024-06-20, v1.16 2024-07-02, v1.17 2024-07-25, v1.18 2024-08-03, v1.19 2024-08-10, v1.20 2024-08-18, v1.21 2024-09-04, v1.22 2024-09-15, v1.23 2024-09-24, v1.24 2024-11-06, v1.25 2024-11-18, v1.26 2024-12-14).
- **No release since 2024-12-14** — roughly 20 months as of 2026-09-02.
- The `main` branch has had no commit since 2024-12-14 (`gh api repos/LLNL/LEAP/commits`).
- Development has continued on branches. Of the 14 branches, the most recent activity is:
  - `version_two` — last commit 2026-07-25, "remove files", preceded by 2026-07-25 "speed improvements, bug fixes, and incorporate XrayPhysics". Its `setup.py` declares `version='2.0'`. This is unreleased and unmerged.
  - `AMD` — last commit 2025-05-16, "update kernensl for AMD (no texture memory GPU)".
  - `champley_dev` — last commit 2025-01-10, "minor fixes" (adds an inpainting algorithm).
  - `system_matrix` — last commit 2024-10-26 ("added capability for cone-beam").
- The repository's `pushed_at` is 2026-07-25T22:59:07Z and `updated_at` is 2026-08-25T08:28:18Z, so the project is not dormant, but the released version has been static since December 2024.
- Issue traffic is live through mid-2026, e.g. issue #213 opened 2026-07-31 ("load_tif_python ignores a provided array and crashes when called without one"), #212 (2026-06-17), #210 (2026-06-03), #208 (2026-06-03, "New Conda-Forge Package").
- Maintainer responsiveness has thinned: the most recent issue comments by `kylechampley` in the last 20 issue comments are dated 2026-01-23 and 2026-01-22 (`gh api repos/LLNL/LEAP/issues/comments?sort=created&direction=desc`). Comments after that date in that window are from other users.
- Prebuilt binaries are attached to every release, but only for Windows and Linux: `libleapct.dll` and `libleapct.so`. v1.26 download counts are 383 (dll) and 635 (so). v1.15 additionally shipped `libleapct_cuda12.dll`. No macOS `.dylib` has ever been released.
- Repository topics: abel-transform, artificial-intelligence, beamhardening, bhc, computed-tomography, cone-beam, ct, helical-reconstruction, image-reconstruction, iterative-reconstruction, leap, machine-learning, medical-imaging, metal-artifact-reduction, parallel-beam, scatter-correction, spect, tomographic-reconstruction, tomography, vam.
- Citation request in the README: "Please cite our work by referencing this github page and citing our article: Hyojin Kim and Kyle Champley, 'Differentiable Forward Projector for X-ray Computed Tomography', ICML, 2023" (arXiv:2307.05801). A second paper is requested for specific features: "If you use RDLS, azimuthalFilter, or histogramSparsity, please cite the following paper: Champley, Kyle M., Michael B. Zellner, Joseph W. Tringe, and Harry E. Martz Jr. 'Methods for Few-View CT Image Reconstruction.' arXiv preprint arXiv:2410.07552 (2024)." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/README.md).

---

## 16. Stated limitations and caveats

Collected from source strings, docstrings, demos, and issue titles.

### 16.1 Geometry and algorithm scope

- BPF works only for parallel-beam data (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2887).
- `FBP_adjoint` "will not provide an exact adjoint of FBP for fan-beam or helical scans" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2749).
- Attenuated Radon Transform: parallel-beam only for backprojection; analytic inversion only for a 360-degree angular range (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d06_ART.py, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors_attenuated.cu#L878).
- Symmetric-object mode requires exactly one projection angle and an axis within 30 degrees of z, and only for cone or parallel (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/parameters.cpp#L627).
- No analytic circle-plus-line reconstruction (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d24_circle_plus_line.py).
- Modular-beam FBP requires the detector column vectors to be within 5 degrees of the z axis; otherwise only iterative reconstruction is possible (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d04_modularbeam.py, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1188).
- `find_centerCol`, `find_tau`, `estimate_tilt`, `conjugate_difference` do not work for modular-beam, and give poor estimates on truncated projections (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L846, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L892, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L937, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L976; C++ guard at https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/find_center_cpu.cpp#L352).
- `consistency_cost` works only for axial flat-panel cone-beam and untruncated projections (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1021).
- Extended-SF gaps: "Error: cone-parallel projector not yet implemented for small voxels" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors_extendedSF.cu#L2018) and "Error: cone-parallel backprojector not yet implemented for large voxels" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors_extendedSF.cu#L2124).
- Only SF and VD projectors are selectable; Siddon and Joseph are not user-selectable (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L1930).
- Volume masking "does not improve the speed of forward or backprojection algorithms" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3378).

### 16.2 CPU-only or GPU-only restrictions

- Multi-GPU requires ZYX volume order and CPU-resident input; GPU-resident torch tensors use a single GPU (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1630, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L737, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d02_standard_geometries_torch.py).
- Rebinning routines are CPU-only (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d10_fan_to_parallel_rebinning_and_cone-parallel.py).
- Geometric calibration metrics require CPU-resident data (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L3932-L3965).
- 15 functions in `tomographic_models.cpp` are GPU-only (see 10.2).
- CPU projectors reject voxel sizes far from nominal (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors.cpp#L92-L119).
- `beam_hardening_heel_effect` and `convertToRhoeZe` are not implemented for torch tensors (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L4781, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L4878).

### 16.3 Physics limits

- Scatter model is first-order only, single-material (variable density), and requires heavy down-sampling: volume ≤ 200^3, projections ≤ 256^2, spectra ≤ 20 samples (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1473).
- Gain correction supports exactly 3 calibration scans (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L150).
- BHC beyond two materials, triple-energy decomposition and spectral calibration are listed as future work (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/README.md).
- The supplied denoising dictionaries are complete, not overcomplete: "We provide some sample dictionaries... The dictionaries we currently supply are complete (not overcomplete) and we are working on good overcomplete dictionaries which we will include in future releases." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d27_dictionaryDenoising.py).
- The spectral calibration demo is described as preliminary: "We will make changes to this demo script in the future to make it more robust and run faster. For now, we just want to provide a basic working example." (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d37_spectral_calibration.py).
- The TIGRE bridge is unverified: "I do not gaurantee that this works correctly in all cases and some features may be missing!!!" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/utils/tigre_geometry_bridge.py).
- The LTT bridge is incomplete: "Some parameters mappings are not yet implemented, such as non equi-spaced projection angles" (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/utils/bridgeToLTT.py).

### 16.4 Code-level FIXMEs and comments in the leaptorch path

- `lct.fbp_adjoint_gpu(g, f) # compute proj (g) from input (f) -> needs to be replaced!!!` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leaptorch.py#L139, and https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leaptorch.py#L120 for CPU).
- `// FIXME: this does not properly calculate the amount of memory necessary` in `project_multiGPU_splitViews` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L983).

### 16.5 Selected open issues (title and open date) that bear on capability

From `gh issue list --repo LLNL/LEAP --state open`:

- #213 (2026-07-31) "load_tif_python ignores a provided array and crashes when called without one"
- #212 (2026-06-17) "Correspondence between the z-axis coordinates of CT projection data and CT volume"
- #210 (2026-06-03) "rowvector of modularBeam"
- #208 (2026-06-03) "New Conda-Forge Package"
- #205 (2026-02-02) "error in projection with modular geometry"
- #200 (2026-01-19) "project() function exits for certain geometries."
- #196 (2025-12-02) "segmented artifacts along the Z-axis with a periodicity equal to the pitch in helical CT"
- #192 (2025-11-03) "z-slices outside detected FOV should be zero, but are repeats of last valid slice"
- #191 (2025-10-18) "long object artifact"
- #188 (2025-09-25) "Compilation with cpu_CMakeLists.txt fails"
- #183 (2025-07-11) walnut dataset from Zenodo — "the reconstruction quality degrades significantly"
- #170 (2025-05-24) "cuda memory leak"
- #169 (2025-05-21) "Unable to install LEAP"
- #166 (2025-05-05) "FBPFunctionGPU in leaptorch"
- #157 (2025-03-18) "Have issue when using RDLS to perform limited-angle cone beam reconstruction"
- #143 (2024-12-26) "leaptorch Projector's bp and fbp error"
- #138 (2024-11-21) "Moiré Pattern in FBP reconstruction"
- #127 (2024-10-27) "Build error linking leapct libraries to python site-packages on Windows 10"
- #125 (2024-10-27) "Increasing GPU memory usage caused by the iterative use of FBP"

---

## 17. Installation, from the wiki

- Standard install: `pip install .` from a clone; "It is strongly recommended to run 'pip uninstall leapct' if you have installed the previous version." (https://github.com/LLNL/LEAP/wiki/Installing-and-Using-LEAP). Note this is a source build, not a wheel install: https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/setup.py shells out to `sh ./etc/build.sh` (Linux/macOS) or `.\etc\win_build_agn.bat` (Windows) before calling `setup()`.
- PyTorch-free install: rename `setup_ctype.py` to `setup.py` and `pip install .`; only numpy is then required (https://github.com/LLNL/LEAP/wiki/Installing-LEAP-without-PyTorch).
- CMake install: `sh ./etc/build.sh` on Linux/macOS, `.\etc\win_build.bat` on Windows (which produces a Visual Studio solution at `LEAP\win_build\leap.sln`).
- Precompiled dynamic libraries: download `libleapct.dll` or `libleapct.so` from the releases page, then either run `python manual_install.py` (copies the library and the four Python files into site-packages) or add the `src` folder to `PYTHONPATH` (https://github.com/LLNL/LEAP/wiki/Using-the-LEAP-precompiled-dynamic-libraries, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/manual_install.py).
- The four Python modules installed are `leapctype.py`, `leaptorch.py`, `leap_filter_sequence.py`, `leap_preprocessing_algorithms.py` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/setup.py, `py_modules`).
- HPC-specific instructions for Livermore Computing (Intel/Linux and IBM PowerAI) load `gcc/8.3.0` and `cuda/11.7.0` (https://github.com/LLNL/LEAP/wiki/Installing-and-Using-LEAP).
- The wiki and README describe no binary-package install. There is no PyPI package; a community conda-forge package exists. Both are covered in section 20.

---

## 18. Publications

### 18.1 The LEAP paper: Kim and Champley, ICML 2023 workshop (arXiv:2307.05801)

- Source: https://arxiv.org/abs/2307.05801 ; full text at https://arxiv.org/pdf/2307.05801 and https://arxiv.org/html/2307.05801v1
- Title: "Differentiable Forward Projector for X-ray Computed Tomography". Authors Hyojin Kim and Kyle Champley, marked "*Equal contribution". Champley is listed as "Ziteo Medical, California, USA, He conducted the software development and experiments while affiliated with Lawrence Livermore National Laboratory."
- Submitted 11 July 2023. Venue line: "Published at the Differentiable Almost Everything Workshop of the 40th International Conference on Machine Learning, Honolulu, Hawaii, USA. July 2023."
- DOI https://doi.org/10.48550/arXiv.2307.05801 ; report number LLNL-CONF-849839.
- Length: 5 pages (4 pages of body plus references). This is a workshop extended abstract, not a full journal paper.
- Abstract, verbatim: "Data-driven deep learning has been successfully applied to various computed tomographic reconstruction problems. The deep inference models may outperform existing analytical and iterative algorithms, especially in ill-posed CT reconstruction. However, those methods often predict images that do not agree with the measured projection data. This paper presents an accurate differentiable forward and back projection software library to ensure the consistency between the predicted images and the original measurements. The software library efficiently supports various projection geometry types while minimizing the GPU memory footprint requirement, which facilitates seamless integration with existing deep learning training and inference pipelines. The proposed software is available as open source: https://github.com/LLNL/LEAP."
- Projector models described in the paper: "We chose to implement the Siddon and SF projector methods in our software package." SF is cited to Long, Fessler and Balter, TMI 2010. Rationale: "The second two of these methods, DD and SF, model the finite width of the detector pixels and volume voxels, while the first two of these methods do not." On adjoints: "Since our goal here is to implement methods that are stable over a thousand or more iterations, we chose to implement methods where the exact transpose is used."
- Geometry coverage at the time of writing (July 2023): "parallel-beam, axial cone-beam (planar or curved detector), and a flexible cone-beam geometry", with "future releases will include fan-beam and helical cone-beam geometries." (Both are now present in v1.26.)
- Implementation note: "Our CUDA implementation utilizes 3D threads and 3D texture memory is used for the input data."

Two experiments only:

1. Data consistency with an inference model, on the ALERT TO4 airport-luggage dataset (Northeastern University): "split into 165 bags for training and the remaining 25 bags for test. The image dimension is 512^2 and the number of projections is 720 (parallel beam). To demonstrate the limited-angle CT, we randomly masked 120 degrees out of 180 degrees (60 degrees available)." The network is "a neural network model combining CT-Net (Anirudh et al., 2018) and U-Net (Han and Ye, 2018)."
2. Timing and memory of forward projection versus LTT.

**Accuracy number** (limited-angle experiment), verbatim: "The refinement step with our projector led to an improvement in the averaged PSNR (dB) and SSIM from 35.486 and 0.905 to 36.350 and 0.911, respectively." The single example in Figure 3 is labelled 29.453/0.816 before and 30.708/0.841 after the data-consistency step.

**Timing and memory — Table 1 of the paper, values verbatim.** Caption: "Table 1: Performance comparison (sec) between ours and LTT. (.) indicates the memory usage (GB). In our method, we report the times without and after the CPU-GPU data transfer. The dimension refers to the image dimension and the number of projections."

| Geometry / size | Parallel 512^3 / 180 | Parallel 1024^3 / 720 | Cone 512^3 / 180 | Cone 1024^3 / 720 |
| --- | --- | --- | --- | --- |
| Ours (LEAP) | 0.5 / 1.8 (1.5) | 11.5 / 15.4 (8) | 1.4 / 2.8 (1.5) | 37.1 / 39.2 (11.1) |
| LTT | 4.2 (-) | 17.4 (-) | 4.5 (-) | 38.9 (-) |

Hardware and conditions, verbatim: "on NVIDIA Tesla P100 with 16GB"; "The angular ranges for parallel and cone beams are 180 and 360, respectively"; "Note that we did not list the memory usage for LTT because it is a user-specified parameter." **No CPU model is stated. No detector dimensions are given beyond the image dimension. The times are for forward projection only.**

**Comparison scope.** The only quantitative comparison in the paper is against **LTT (Livermore Tomography Tools)**, LLNL's own earlier, closed-source package (Champley et al., NDT & E International 126:102595, 2022, doi 10.1016/j.ndteint.2021.102595). There is **no comparison to ASTRA, TIGRE, torch-radon, tomosipo, or ODL** in this paper. ASTRA appears only as a citation in the sentence: "most reconstruction packages (Aarle et al., 2015) violate this requirement because exact transposes are typically not as computationally efficient as other methods." TIGRE, torch-radon, tomosipo and ODL are not mentioned or cited.

### 18.2 Champley, Zellner, Tringe, Martz Jr., "Methods for Few-View CT Image Reconstruction" (arXiv:2410.07552)

- Source: https://arxiv.org/abs/2410.07552 ; full text https://arxiv.org/pdf/2410.07552v1 (14 pages). Submitted 10 October 2024. DOI https://doi.org/10.48550/arXiv.2410.07552. No journal reference. Subjects physics.med-ph, cs.MS, physics.comp-ph.
- Abstract, key part verbatim: "Here we develop constrained and regularized numerical optimization methods to reconstruct CT volumes from 4-28 projections... The efficacy of our methods is demonstrated on four measured and three simulated few-view CT data sets. We show that these methods outperform other state of the art few-view numerical optimization methods."
- Four new algorithms named in Section 7: "Simple Function Least Squares reconstruction algorithm (Section 5.2)", "Regularized Derivative Least Squares (Section 5.3)", "Histogram Sparsity Regularization (Section 5.4)", "Azimuthal Sparsity Regularization (Section 5.5)". Section 5.6 gives a space-carving foreground estimate `f_FG := 1 - u(P^T(1 - g_FG))`, which is LEAP's `space_carving`.
- RDLS in the paper: `Phi_RDLS(f) := 0.5*[grad(Pf-g)]^T[grad(Pf-g)] + beta*R(f)`, equivalently `-0.5*(Pf-g)^T Laplacian (Pf-g) + beta*R(f)`. Observation given: "Note that -Phi'_RDLS(0) = -P^T Laplacian g which is roughly equal to a Lambda or local tomography reconstruction." Caveat given: "low frequency image features converge very slowly."
- Histogram sparsity: `Psi(f;mu) := alpha*||u_mu(f)||_1` with `u_mu = prod_k (f-mu_k)^2/((f-mu_k)^2 + 0.5*m^2)`, `m := min_k |mu_k - mu_{k+1}|`. Example uses "target values mu = (0.0, 0.02, 0.069)".
- Azimuthal sparsity: `Psi(f;phi) := alpha*||h(v_phi(f) - f)||_1` where `v_phi` is an azimuthal low-pass filter. "We have found that a moving average filter works well."
- **The paper has no results tables — no RMSE, PSNR, SSIM, or timing values anywhere in its 14 pages.** The evaluation is visual. The only numbers are experiment descriptions: "a 180 um diameter glass fiber with 19 projections" (Xradia Ultra XRM at LLNL); the DEVCOM ARL MEFCT system with "five 150 kV source-detector pairs, five 300 kV source-detector pairs, and five 450 kV source-detector pairs"; "we simulated 28 fan beam projections"; "we simulated 24 fan beam projections of the FORBILD head phantom"; an ALS synchrotron foam sample where "we use only 15 of the 1440 projections"; "simulated 15 noisy cone-beam projections of a two-layer pipe phantom" with an air gap that "spans 70 degrees around the pipe", FBP post-processed with "a 45 degree azimuthal blur" versus RWLS with azimuthal sparsity where "blur kernel is 4 degrees in size". One stability observation: "applying ADS-POCS diverged on the second iteration."
- **Attribution caveat, verbatim: "All numerical experiments were performed using the Livermore Tomography Tools (LTT) software package [28]. These algorithms can also be found in the LEAP-CT [29] open source library as well."** The results in this paper were therefore not produced with LEAP.

### 18.3 No journal paper describing LEAP

- Searches for a SoftwareX, JOSS, Optics Express, NIM, or IEEE TCI paper, and for the phrase "LivermorE AI Projector", found no peer-reviewed journal article describing LEAP. The repository asks only for the ICML workshop paper (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/README.md).
- There is a DOE CODE software record: https://www.osti.gov/doecode/biblio/108702 — "LivermorE AI Projector for Computed Tomography Tasks", Kim, Hyojin; Champley, Kyle M.; release date October 7, 2022; DOI https://doi.org/10.11578/dc.20230622.3; software version 1.0; MIT; LLNL-CODE-848657; contract AC52-07NA27344 (NNSA).
- A related Champley paper on symmetric objects, "Tomographic Model Based Iterative Reconstruction of Symmetric Objects" (https://arxiv.org/html/2410.09837v1), states that both LTT and LEAP-CT implement the tilted symmetric X-ray Transform.

### 18.4 Citation counts

- OpenAlex, retrieved 2026-09-02 from https://api.openalex.org/works/doi:10.48550/arXiv.2307.05801 (work https://openalex.org/W4384263560, type `preprint`): **`cited_by_count: 5`**. OpenAlex finds only one work matching the title, so there is no separately indexed published version splitting the count. The five citing works, all 2025: "Artificial Intelligence in Computed Tomography Image Reconstruction: A Review of Recent Advance...", "Real-time volumetric CBCT reconstruction using surface and X-ray imaging for image-guided radio...", "Task based evaluation of sparse view CT reconstruction techniques for intracranial hemorrhage d...", "XCal: model-based approach to X-ray CT spectral calibration", "Near-isotropic super-resolution CBCT imaging with a dual-layer flat panel detector".
- Semantic Scholar for arXiv:2410.07552 (https://api.semanticscholar.org/graph/v1/paper/arXiv:2410.07552): `"citationCount": 3, "influentialCitationCount": 0`.
- Semantic Scholar was rate-limited (HTTP 429) for arXiv:2307.05801 across repeated attempts, and Google Scholar was not accessible, so the only count available for the main LEAP paper is OpenAlex's 5.

---

## 19. Third-party benchmarks

**No published third-party benchmark comparing LEAP against another CT toolkit was found.** Two recent multi-toolkit benchmark papers were checked and neither includes LEAP:

- "Benchmarking Open-Source FDK Against Commercial and Iterative Reconstruction Methods for Preclinical Micro-CBCT" (https://arxiv.org/html/2604.23047v1) — compares an in-house FDK, GE vendor software, ASTRA SIRT, and TIGRE OS-SART on an NVIDIA H100. LEAP, LLNL, and Champley are not mentioned.
- "Benchmarking learned algorithms for computed tomography image reconstruction tasks" (https://arxiv.org/html/2412.08350v1) — uses LION, tomosipo, ASTRA; mentions ODL and TIGRE. LEAP is not mentioned.

An additional self-published claim found off-repository, stronger than the README's wording:

- LLNL software portal, https://software.llnl.gov/news/2024/01/07/leap-new/, verbatim: "LEAP contains more accurate projectors and FBP algorithms, more features, and most algorithms run faster than other popular CT reconstruction packages." Note this says "run faster" where the README says "run as fast or faster".

**On the 1.7x SNR claim.** The README's "The LEAP reconstruction has 1.7 times higher SNR than ASTRA" is not reproducible from public information: no ASTRA version, hardware, walnut-dataset citation, reconstruction parameters, SNR definition, or ROI is published for that figure. The explainer the README links to (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/results/SF_vs_VD.md) reports a *different* experiment — a simulated SF-versus-VD ablation inside LEAP with over-sized voxels — giving "SNR for the VD result is 25.7 and the SNR for the SF result is 43.6" (ratio 1.70). That document itself connects the two: "This is the reason why the LEAP result of the walnut reconstruction on the LEAP readme page had higher signal to noise ratio." The mechanism it describes is that separable-footprint backprojection with voxels larger than nominal averages over more detector pixels than bilinear-interpolation voxel-driven backprojection, i.e. it is a resolution-for-noise trade rather than a free improvement; the same document states that VD gives higher resolution in that regime and is about twice as fast.

**No speed benchmarks were found on the LEAP GitHub wiki or in its Discussions.**

---

## 20. Packaging: PyPI, conda-forge, binaries

### 20.1 PyPI — no LEAP package exists

Verified against the PyPI JSON API on 2026-09-02:

- https://pypi.org/pypi/leapct/json -> HTTP 404, `{"message": "Not Found"}`
- https://pypi.org/pypi/leap-ct/json -> HTTP 404
- https://pypi.org/pypi/leap/json -> HTTP 200, but this is an unrelated package (`leap` 2021.1, "Time integration by code generation", Andreas Kloeckner, https://documen.tician.de/leap)
- https://pypi.org/pypi/xrayphysics/json -> HTTP 404 (the companion package is also absent from PyPI)

There are **no wheels of any kind and no PyPI presence**. `pip install .` in the wiki means compiling from a local source checkout (see section 17). The README contains none of the strings "conda", "pip install", "wheel", or "pypi".

### 20.2 conda-forge — leapct 1.26 exists

- https://api.anaconda.org/package/conda-forge/leapct and recipe https://raw.githubusercontent.com/conda-forge/leapct-feedstock/main/recipe/recipe.yaml
- `conda-forge/leapct` **version 1.26**, `noarch: python`, MIT, 966 downloads. Feedstock created 2026-05-13T17:00:34Z, last pushed 2026-05-14. Package page http://anaconda.org/conda-forge/leapct
- Run dependencies, verbatim from the recipe: `python >=3.10`, `numpy`, `scipy`, `matplotlib-base`, `imageio`, `pytorch * cuda*`, `xrayphysics`, `cuda-version >=12.0`, `libleapct ==1.26 he44769d_0`.
- `conda-forge/libleapct` 1.26 holds the compiled library. Its platforms are **linux-64 and win-64 only — no osx-64 and no osx-arm64.**
- The recipe skips non-CUDA builds: `build: skip: - cuda_compiler_version == "None"`. There is therefore **no CPU-only and no macOS conda package**, and the PyTorch dependency is pinned to a CUDA build.
- The recipe carries a downstream-only patch, described in the recipe as: "This patch is not upstream... overwrite the default search directory for the leapct shared library... without the patch, dll lookup fails and the library is unable to function."
- `conda-forge/xrayphysics` exists at version 1.2.1, noarch, MIT.
- This is a community packaging effort: conda is never mentioned in the LEAP README or wiki, and there is an open LEAP issue #208 (2026-06-03) titled "New Conda-Forge Package".

### 20.3 Prebuilt binaries

- Every release from v1.9 to v1.26 attaches exactly `libleapct.dll` and `libleapct.so`; v1.15 additionally shipped `libleapct_cuda12.dll`; v1.0 through v1.8 shipped `libleap.dll` and `libleap.so`. **No macOS binary has ever been released** (GitHub releases API).
- The wiki frames binaries as a fallback: "We recommend using the LEAP pip install method outlined above, but if you are still having issues installing LEAP, you can use the precompiled dynamic libraries." (https://github.com/LLNL/LEAP/wiki/Using-the-LEAP-precompiled-dynamic-libraries).

---

## 21. readthedocs

- https://leapct.readthedocs.io/ , project slug `leapct`, version slug `latest`.
- **The site displays "LEAP v1.4".** The page `<title>` is "LivermorE AI Projector for Computed Tomography — LEAP v1.4 documentation", because https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/conf.py#L38 contains `release = 'v1.4'` and has never been updated. The package is at 1.26, so the documented version string is 22 releases stale. The content itself is built from `main`, so it is current for v1.26 even though the label is not.
- Sphinx extensions used: `['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx_rtd_theme']` (https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/conf.py#L55).
- Top-level table of contents: Installation and Tutorial; Technical Manual (an external link to the PDF on GitHub, not a docs page); Python API (leapctype); Preprocessing Algorithms; Physics-Based Preprocessing Algorithms; PyTorch Neural Network Interface (leaptorch); Indices and tables.
- Python API sub-pages: High-Level Functions; CT Geometries; CT Volume; Forward and Back Projection; Analytic Reconstruction (FBP); Iterative Reconstruction; Filter Sequence; Projection Filters and Transforms; Volume Filters and Transforms; Filters Applied to Arbitrary 3D Arrays; File I/O; CT Simulation Functions.
- What the site has that the `.rst` files do not: the rendered API pages are autodoc output, so they carry the full signatures and docstrings pulled live from `leapctype.py`, `leaptorch.py`, `leap_filter_sequence.py` and `leap_preprocessing_algorithms.py`. The leaptorch page enumerates `BackProjectorFunctionCPU, BackProjectorFunctionGPU, BaseProjector, FBP, FBPFunctionCPU, FBPFunctionGPU, FBPReverseFunctionCPU, FBPReverseFunctionGPU, Projector, ProjectorFunctionCPU, ProjectorFunctionGPU`. Plus the generated `genindex.html`, `py-modindex.html`, `search.html`.
- The Installation page (https://leapct.readthedocs.io/en/latest/install.html) is a five-line stub that only links out to GitHub, the releases page, the wiki, and the two demo directories.
- The technical manual is not rendered on readthedocs; the nav entry is an external link to https://github.com/LLNL/LEAP/blob/main/documentation/LEAP.pdf (902,180 bytes). Its LaTeX title is `\title{LEAP Technical Manual \\ Version 1.1}`, so the manual also carries a stale version label.

---

## 22. Companion projects

### XrayPhysics

- https://github.com/kylechampley/XrayPhysics — 55 stars, MIT, last pushed 2025-01-19. Docs at https://xrayphysics.readthedocs.io/
- GitHub description: "C/C++ library and Python bindings for x-ray cross sections, x-ray source spectra modeling, multi-material beam hardening correction, and dual energy decomposition algorithms."
- Capabilities per its README: x-ray cross sections from 1 keV to 20 MeV for elements 1-100, by chemical formula or mass fractions; incoherent and coherent scattering angle distributions; source spectrum models across voltages, take-off angles and anode materials (Cu, Mo, W, Au); beam hardening correction (theoretical and polynomial); effective atomic number by LLNL definitions; dual energy decomposition. Data derives from EPDL97, and "The cross section tables are hard-coded into C++ arrays, so queries of the database are instant."
- Relationship to LEAP: "Designed to be used in conjuction with (but is not required) LEAP", and it enables LEAP's SIRZ.
- On conda-forge as `xrayphysics` 1.2.1 (noarch); not on PyPI. Note that conda-forge's `leapct` lists `xrayphysics` as a hard run dependency even though the README calls it optional.

### LEAPCT-UI-GUI

- https://github.com/kylechampley/LEAPCT-UI-GUI — 24 stars, MIT, created 2024-06-12, last pushed 2024-11-08.
- GitHub description: "high level UI and GUI for the LEAP-CT library." README: "a high level UI and GUI for the LEAP-CT and XrayPhysics libraries", split into `leapctserver.py` (backend UI) and "LEAP-CT Rails" (a PyQt5 GUI).
- It handles file I/O, drives LEAP algorithms "with less flexibility but more ease of use", and manages memory **by using hard-drive storage when CPU RAM is insufficient** (data chunking) — this is the only out-of-core / host-memory-overflow mechanism in the LEAP ecosystem, and it lives in the GUI project rather than in LEAP itself.
- Requirements: LEAP-CT v1.21 or newer, Python 3.10 or newer, PyQt5; XrayPhysics recommended but optional. Install by cloning and `pip install .`, which generates a desktop launcher.

---

## 23. Summary of every performance number found, with its source and hardware

| Number | Exact claim | Hardware / conditions | Source |
| --- | --- | --- | --- |
| 0.5 / 1.8 s, 1.5 GB | LEAP forward projection, parallel beam, 512^3 image, 180 projections, 180-degree range; "times without and after the CPU-GPU data transfer" | NVIDIA Tesla P100 16 GB; no CPU stated | arXiv:2307.05801 Table 1 |
| 11.5 / 15.4 s, 8 GB | LEAP forward projection, parallel beam, 1024^3 image, 720 projections | NVIDIA Tesla P100 16 GB | arXiv:2307.05801 Table 1 |
| 1.4 / 2.8 s, 1.5 GB | LEAP forward projection, cone beam, 512^3 image, 180 projections, 360-degree range | NVIDIA Tesla P100 16 GB | arXiv:2307.05801 Table 1 |
| 37.1 / 39.2 s, 11.1 GB | LEAP forward projection, cone beam, 1024^3 image, 720 projections | NVIDIA Tesla P100 16 GB | arXiv:2307.05801 Table 1 |
| 4.2 s | LTT, parallel beam 512^3 / 180 | same machine; LTT memory "a user-specified parameter" | arXiv:2307.05801 Table 1 |
| 17.4 s | LTT, parallel beam 1024^3 / 720 | same | arXiv:2307.05801 Table 1 |
| 4.5 s | LTT, cone beam 512^3 / 180 | same | arXiv:2307.05801 Table 1 |
| 38.9 s | LTT, cone beam 1024^3 / 720 | same | arXiv:2307.05801 Table 1 |
| 35.486 -> 36.350 PSNR (dB), 0.905 -> 0.911 SSIM | averaged improvement from the data-consistency refinement step | ALERT TO4 luggage data, 512^2, 720 parallel projections, 60 of 180 degrees available, 25 test bags | arXiv:2307.05801 |
| 1.7x | "The LEAP reconstruction has 1.7 times higher SNR than ASTRA" (walnut FDK) | **no hardware, no ASTRA version, no parameters, no SNR definition published** | https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/README.md |
| 25.7 vs 43.6 | SNR of the VD result vs the SF result, voxel size = 2.0 * sod/sdd * pixelWidth, with noise added | **no hardware; simulation, script `d98_SF_vs_VD.py`** | https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/results/SF_vs_VD.md |
| ~2x | "VD backprojection of cone-beam data is about twice as fast as SF-based backprojection in LEAP" | **no hardware, no problem size** | https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/results/SF_vs_VD.md |
| 2-8x | dictionary denoising "runs about 2-8 times faster" when the dictionary is complete and orthonormal | **no hardware** | https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d27_dictionaryDenoising.py |
| "slightly faster" | LEAP's texture-memory SF vs the original rectangle-rectangle SF, "but are very, very slightly less accurate" | **no number, no hardware** | https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex |
| "as fast or faster" | vs "other popular CT reconstruction packages" | **no benchmark** | https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/README.md, https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/LEAP_features.md |
| "run faster" | vs "other popular CT reconstruction packages" | **no benchmark** | https://software.llnl.gov/news/2024/01/07/leap-new/ |

Points worth carrying into a comparison:

- The only published LEAP timings are **forward projection only**, on a **2016-era Tesla P100**, against **LTT** (LLNL's own closed-source package), in a **5-page workshop abstract**. There are no published backprojection, FBP, or iterative-reconstruction timings, and no published multi-GPU scaling numbers despite multi-GPU being a headline feature.
- The 1.7x-SNR-vs-ASTRA claim is an image comparison whose supporting document describes a resolution-for-noise trade at over-sized voxels, not a same-resolution accuracy win.
- The repository ships a benchmark script (`d99_speedTest.py`) but no recorded results.

---

## 24. Uncertain or unverified

Facts I could not confirm, and places where the evidence is thin:

1. **Semantic Scholar / Google Scholar citation count for arXiv:2307.05801.** Semantic Scholar returned HTTP 429 on every attempt and Google Scholar was not reachable. The only count obtained is OpenAlex's `cited_by_count: 5` (retrieved 2026-09-02).
2. **Whether the `version_two` branch (version 2.0, last commit 2026-07-25, "speed improvements, bug fixes, and incorporate XrayPhysics") will be released, and what it changes.** I did not diff the branch against main beyond reading its `setup.py` version string. Any statement about LEAP 2.0's features or speed would be speculation.
3. **Whether v1.26 is still what a user gets today.** It is what GitHub releases, what the source build produces from `main`, and what conda-forge packages — but if a user builds from `version_two` they get different code.
4. **The exact provenance of the "1.7x higher SNR than ASTRA" figure.** No script, ASTRA version, hardware, dataset DOI, or SNR definition is published for it. My inference that it derives from the 43.6/25.7 SF-vs-VD ratio is supported by the sentence in `SF_vs_VD.md` linking the two, but the README figure and the `SF_vs_VD.md` numbers are from different experiments and I could not verify they are the same measurement.
5. **CPU performance.** No CPU timing appears anywhere — not in the repository, not in the papers, not on the wiki. The claim "multi-core CPU implementations of all algorithms that are as fast or faster than other popular CT reconstruction packages" is unmeasured, and the source shows several algorithms have no CPU implementation at all.
6. **Multi-GPU scaling.** Claimed but never measured in any public document.
7. **Apple Silicon.** Nothing in the repository, wiki, papers, or packaging mentions arm64, Apple Silicon, Metal, or MPS. The README's macOS support means a CPU-only source build with gcc after manually swapping in `cpu_CMakeLists.txt`. I did not attempt a macOS build, so I cannot confirm whether that build currently succeeds. Open issue #188 (2025-09-25) "Compilation with cpu_CMakeLists.txt fails" suggests the CPU build path may be broken as of that date; I did not read the issue body or its resolution.
8. **AMD GPU support.** An `AMD` branch exists (last commit 2025-05-16, "update kernensl for AMD (no texture memory GPU)"), `setup_AMD.py` and `hip_utils.h` are on main, but the README lists AMD support as future work. I could not determine whether the AMD branch works.
9. **Whether the `unit_tests.py` script is expected to run as checked in.** It has `geometries = []` immediately before its main loop, which makes the loop a no-op. I did not run it.
10. **DICOM.** I found no DICOM reader or writer, and no vendor-format reader. I searched `src/*.py` and https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/file_io.cpp for "dicom" and found nothing, but I did not exhaustively read `file_io.cpp`.
11. **Whether iterative reconstruction algorithms work correctly for GPU-resident torch tensors in all cases.** The source contains commented-out guards reading "ERROR: Iterative reconstruction algorithms not implemented for torch tensors!" at https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3635, 3710, 3823, 3938, 4589. They are commented out, and demo `d02_standard_geometries_torch.py` exercises this path, so the intent is that it works — but the commented-out guards suggest it was once restricted and I did not test it.
12. **The relationship between the manual's TV weight `||i-j||` and the docstring's `||i-j||^{-1}`.** These are written differently in https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex and https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5385; I did not read the CUDA kernel in `total_variation.cu` to determine which is implemented.
13. **The default `p` exponent for TV.** The Python default is `p=1.2` (`TV.__init__`, `TVcost`), and the manual writes the Huber tail as `(delta^2/6)[5|t/delta|^1.2 - 2]`, consistent with p=1.2, but I did not verify the kernel.
14. **Download and adoption figures.** Release-asset download counts for v1.26 are 383 (`libleapct.dll`) and 635 (`libleapct.so`); conda-forge reports 966 downloads for `leapct`. There is no PyPI download signal because there is no PyPI package. These are the only adoption numbers available.
15. **Maintainer status.** The last issue comment by `kylechampley` within the 20 most recent issue comments is dated 2026-01-23. I checked only the most recent 20 comments, so this is a lower bound on responsiveness, not proof of abandonment; the `version_two` branch commit on 2026-07-25 shows the maintainer is still working on the code.
