# Kerbside

Traffic analytics from a **fixed roadside camera** using **only classical
digital image processing**. From nothing but the pixels and a one-time
geometric calibration, Kerbside recovers how many vehicles passed, in which
lane, in which direction, how fast, what size class and what colour.

No detector. No tracker-by-detection. No learned model anywhere in the pipeline.
This is what deployed traffic counting and speed measurement systems did for two
decades before deep learning, and what many still do.

> **Build status: Phase 2 of 8 complete.** The synthetic renderer, the geometry
> and ground-truth contract, and the operator library are finished and tested.
> The pipeline stages, the API and the operations console are not yet built.
> Everything claimed below is backed by a test in `tests/`; nothing is claimed
> for a stage that does not exist yet.

## Limitations — read this first

Kerbside **measures** speed. It is not, and must not be represented as, a speed
enforcement device.

- A legally admissible speed measurement device requires type approval and
  periodic calibration certification against traceable standards. This project
  has neither and makes no claim to either.
- Nothing here constitutes evidence of an offence. Vehicles recorded above a
  configured limit are labelled **"flagged for review"**, never "violation".
- Accuracy depends entirely on a calibration a user performs by hand, on a road
  they must measure themselves. A calibration that is self-consistent can still
  be wrong by a constant factor, and every speed will be wrong with it. The
  two-point distance tool exists to catch precisely that.
- The system does not read, store, or attempt to recognise number plates or
  faces, and no part of the design would allow it to.

## Install and run

Requires Python 3.11 or newer.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

On Linux or macOS use `.venv/bin/python` throughout.

Generate the synthetic clips and their ground truth:

```bash
.venv/Scripts/python.exe -m kerbside.data.make_clips --list
```

```bash
.venv/Scripts/python.exe -m kerbside.data.make_clips --all --duration 16
```

Run the tests:

```bash
.venv/Scripts/python.exe -m pytest -q
```

Clips are build output and are not committed. They regenerate byte-identically
from their seed, so every number in this document is reproducible from a clean
checkout.

## Why a synthetic renderer comes first

Every accuracy figure this project reports is measured against ground truth. Any
ground truth obtained by hand annotation would be both expensive and
approximate — and, worse, it would be *unfalsifiable* about the one thing that
matters most here, which is geometry. So the data comes from a renderer that
projects cuboid vehicles through a **known** pinhole camera onto a road plane
and records, per frame and per vehicle, exactly what it did.

`data/gt/<scenario>.json` records for every vehicle in every frame:

| Field | Meaning |
|---|---|
| `world_pos_m` | centre of the footprint rectangle on the road plane |
| `ground_point_px` | `H · world_pos_m` — the exact projection, in pixels |
| `bbox_px` | axis-aligned box over all eight projected cuboid vertices |
| `bbox_ground_point_px` | bottom-centre of that box — what S6 will estimate |
| `ground_point_bias_px` | the difference between the two |
| `speed_kmph`, `heading_deg` | from the car-following model |
| `lane_id`, `direction` | assigned at spawn, not inferred |
| `vehicle_class`, `footprint_m` | length, width, height and its class bucket |
| `colour_name`, `body_rgb` | as rendered |
| `occlusion_frac`, `visible_area_px` | measured from an id buffer of the frame |

Two ground points are recorded deliberately. `ground_point_px` is the truth the
homography must reproduce. `bbox_ground_point_px` is the observable a real
pipeline actually gets from a silhouette. Their difference is traps 1 and 11
made measurable rather than argued about.

### The renderer is a fair test, not a rigged one

- **Cast shadows are constructed physically.** Each cuboid vertex is projected
  along the sun direction onto the road plane and the convex hull is darkened.
  Darkening is a per-channel multiplication, which scales HSV value and leaves
  hue and saturation alone — the model S5 will test for. A small per-channel
  ambient correction is applied on top, because a shadowed surface is still lit,
  by the sky rather than the sun, so S5's hue and saturation tolerances are
  genuinely exercised instead of trivially satisfied.
- **Occlusion is measured, not assumed.** Silhouettes are rasterised in
  painter's order into an id buffer and each vehicle's visible pixel count is
  compared against its unoccluded count.
- **Texture belongs to the vehicle, not to the image.** Windows, lights, wheels
  and livery are geometry in the vehicle's own frame, so block matching (Unit 8)
  will measure real texture displacement. Some bodies carry no livery at all, so
  the aperture problem of trap 9 is present in the data rather than hypothetical.
- **Stop-and-go emerges from car following**, not from a scripted slowdown, so
  congested clips contain genuine queue formation and full stops — the input
  trap 5 needs.
- **The container frame rate can be made to lie** on purpose, so S1's timestamp
  measurement will be a tested behaviour rather than a claim.

### Scenarios

| Scenario | What it is for |
|---|---|
| `free_flow` | the reference condition every hard gate is stated against |
| `congested` | queues and full stops — background ghosting (trap 5) |
| `night` | headlight pools on the road that behave like objects (trap 12) |
| `rain` | streaks and contrast loss |
| `heavy_occlusion` | near-lane traffic hiding the far lanes |
| `camera_shake` | pole vibration of known amplitude (trap 8) |
| `long_shadows` | shadows bridging adjacent lanes — the S5 headline (trap 2) |
| `motion_blur` | shutter-integrated silhouettes, biasing the box (trap 11) |

## Results so far

All figures below come from `tests/`, on synthetic data with exact ground truth.

| Study | Result |
|---|---|
| Recorded ground point vs projection of recorded world position | agrees to < 1e-6 px over every vehicle-frame (gate: 1 px) |
| Same, under camera shake | agrees to < 1e-6 px |
| From-scratch DLT vs the closed-form camera homography | < 1e-9 relative |
| From-scratch DLT vs `cv2.findHomography` | < 1e-3 px over the road patch |
| Focal length recovered from the plane homography alone | exact to 1e-9 |
| Out-of-plane points from `H` alone vs the full projection matrix | < 1e-6 px |
| Homography from four **exact** clicks | < 1e-9 relative — criterion met |
| Homography from four **half-pixel** clicks, 18–60 m | median and 95th percentile both < 2% — criterion met |
| Homography from four **whole-pixel** clicks, 18–60 m | ~1.2% median, tail past 2% |

## The operator library

Thirty-five operators across syllabus units 2 to 8, each registered with an id,
its syllabus unit, a parameter schema and a one-sentence rationale. That
metadata is not documentation for its own sake: it is what lets the inspector
build a control for an operator it has never seen, what bounds the values the
API will accept, and what "reset to default" resets to. A test asserts the
schema and the function signature cannot drift apart.

Several operators are written out from their definitions rather than called,
because the point of the exercise is that they are understood: two-dimensional
correlation and convolution, the normalised DLT, Otsu's threshold, Canny's four
stages, the Hough accumulator, the Hadamard matrix by Sylvester recursion, HSI
from its trigonometric definition, m-connectivity labelling, and all three block
matching searches. Where a mature implementation exists, the tests check
agreement with it — that is the honest basis for the claim, and it is how the
Canny bug below was found.

### Measured results

| Study | Result |
|---|---|
| Hand-written correlation vs OpenCV, float input | < 1e-3 max absolute difference |
| Canny vs OpenCV on a real frame | IoU 0.86; 0.97 on a synthetic diagonal |
| Otsu vs OpenCV (textbook `argmax` mode) | identical |
| DFT round trip | < 1e-9 |
| Phase correlation, integer shift | < 0.05 px |
| Phase correlation, sub-pixel shift | < 0.1 px |
| Phase correlation under a 1.4x brightness change | < 0.2 px — unaffected |
| Energy compaction at 4 of 64 coefficients | DCT 0.996 > Hadamard 0.994 > DFT 0.991 |
| Block matching, exhaustive, on real content | MV MAE **0.32 px**, **96%** exact |
| Fast searches vs exhaustive | **36x** (2-D log) and **32x** (three-step) fewer comparisons |
| Fast search accuracy cost | MAE 5.2 px, 53-60% exact, prediction PSNR 35 dB against 42 dB |
| Test coverage of `operators/` | 84% |

The block-matching row is the one worth reading twice. The fast searches are
30-odd times cheaper and still predict the frame at 35 dB, which is perfectly
adequate for a video codec — and they land five pixels from the true vector,
which is not adequate for a speed measurement. The same algorithm can be a good
choice for one job and a bad one for the other, and that distinction is the
finding, not a caveat.

### Findings worth stating plainly

**The bounding-box ground point is not the centre of the vehicle.** It maps to
the vehicle's *near extremity* — exactly half a vehicle length short of the
centre, and exactly so for an unrolled camera, because the image row of a road
point depends only on distance along the road. Absolute position therefore
carries a `length / 2` bias. But that bias is constant for a given vehicle, so
it cancels under differencing, which is why an unbiased speed can be built on a
biased position. Counting-line crossings, which do not difference, will need to
account for it.

**The centroid is much worse, and dangerously so.** Its error grows with both
distance and vehicle height — correlation ≈ 0.85 with height on the free-flow
clip. It never presents as a bug. It presents as trucks reading differently from
cars, which is exactly trap 1.

**Calibration accuracy is limited by click precision, not by the estimator.**
The DLT recovers the rendering homography essentially exactly from exact input.
What costs accuracy is the user's mouse, and how far down the road the geometry
is extrapolated: beyond about 60 m a single pixel is worth metres. The
calibration view will therefore offer zoom, recommend a large quadrilateral, and
report the range over which the geometry can be trusted.

**A four-point calibration's reprojection residual is meaningless.** Four points
determine a homography exactly, so the residual is zero by construction whatever
the user clicked, and whatever real dimensions they typed in. The two-point
distance tool and the implied-camera plausibility check are the only things that
catch a wrong scale.

**m-connectivity and 8-connectivity always produce identical components.** This
is provable, and the tests assert it over random masks: if a shared 4-neighbour
of a diagonal pair is foreground then a 4-path already joins them, and if
neither is, m admits the diagonal directly. What m changes is the *uniqueness of
the path* between two pixels, which matters for boundary following and thinning,
not for counting. The connectivity study therefore reports count differences
between 4 and 8, and path ambiguity for m, rather than implying all three differ
in the same way.

**Otsu's criterion is flat across an empty histogram gap.** Every level inside
the gap separates the two modes equally well, so a plain `argmax` returns the
lowest one — right against the upper tail of the dark mode. On a background
difference, where the background is a spike near zero and the foreground a long
tail, that admits far more noise than a level in the middle would. Taking the
midpoint of the plateau is no less optimal by Otsu's own criterion and is
markedly more stable. And a *unimodal* Gaussian already scores 0.64
separability, so any test for bimodality has to sit well above that number
rather than at some small one.

**Sub-pixel phase correlation needs more than a parabolic fit.** For a
fractional shift the inverse transform of a pure phase ramp is a periodic sinc
whose side lobes alternate in sign, so fitting a parabola through the peak and
its neighbours is fitting a parabola to something that is not remotely
parabolic. It lands about a pixel out — and OpenCV's `phaseCorrelate` has the
same problem, measured. Refining on an upsampled DFT grid instead brings it
under a tenth of a pixel.

## Repository layout

```
backend/kerbside/
  core/
    types.py             typed records; units in every identifier
    registry.py          operator ids, units, parameter schemas, rationales
  operators/
    point.py             negative, stretch, threshold, log, gamma, equalisation
    spatial.py           convolution from scratch, box, gaussian, median, sharpen
    frequency.py         DFT, filters, notch, phase correlation
    transforms.py        DCT, Hadamard, compression and energy-compaction study
    morphology.py        erode/dilate/open/close, hit-or-miss, thinning, banded
    edges.py             Roberts, Prewitt, Sobel, LoG, Canny, Hough
    segment.py           Otsu, adaptive, 4/8/m connectivity, region growing
    colour.py            HSV/HSI/YCbCr/normalised rgb/c1c2c3, shadow, naming
    motion.py            SAD/SSD/MAD, three searches, fields, I/P classification
  geometry/
    homography.py        normalised DLT written from scratch, camera geometry
    calibration.py       four-point calibration, validation, lane definition
    perspective.py       expected size and SE size as a function of image row
  data/
    renderer.py          synthetic scene renderer with exact ground truth
    scenarios.py         presets, each targeting a specific failure mode
    degrade.py           resolution, JPEG, blur, noise, low light
    make_clips.py        CLI: render clips and ground truth to disk
tests/                   pytest; every claim above is asserted here
```

Not yet built: `stages/`, `eval/`, `api/`, and the frontend.

## Syllabus mapping

The full mapping table lands with the pipeline. What Phase 1 establishes:

| Unit | Where it appears so far |
|---|---|
| 1 | Image formation and coordinates; the perspective scale tables traps 3 and 4 depend on; resolution and grey-level degradation; city-block vs Euclidean gradient magnitude |
| 2 | Point processing and neighbourhood filtering, including convolution written from its definition; frequency-domain filtering |
| 3 | DFT and spectra; phase correlation; DCT, Hadamard and the compression study |
| 4 | Erosion, dilation, opening, closing, hit-or-miss, thinning, reconstruction, and perspective-banded morphology |
| 5 | Roberts, Prewitt, Sobel, Laplacian, LoG and Canny, plus the Hough transform |
| 6 | Otsu, adaptive thresholding, 4/8/m connectivity, region growing, split-and-merge, watershed |
| 7 | Six colour models, circular hue arithmetic, shadow discrimination and colour naming |
| 8 | SAD/SSD/MAD costs, exhaustive/2-D log/three-step search, motion fields, compensation and I/P classification |

## Conventions

Every physical quantity carries its unit in the identifier: `speed_kmph`,
`dist_m`, `dx_px`, `dt_s`, `elapsed_ms`. World frame is +X across the road, +Y
along it away from the camera, +Z up, with the road surface at Z = 0. Image
frame is x right, y down, origin top-left. Homographies are stored normalised so
that `h[2, 2] == 1`.
