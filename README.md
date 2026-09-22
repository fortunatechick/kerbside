# Kerbside

Traffic analytics from a **fixed roadside camera** using **only classical
digital image processing**. From nothing but the pixels and a one-time
geometric calibration, Kerbside recovers how many vehicles passed, in which
lane, in which direction, how fast, what size class and what colour.

No detector. No tracker-by-detection. No learned model anywhere in the pipeline.
This is what deployed traffic counting and speed measurement systems did for two
decades before deep learning, and what many still do.

> **Build status: complete, with Phase 3 below target and said so.**
> All nine stages, the REST/MJPEG/WebSocket API, the operations console, the
> nine ablation studies and the self-contained HTML report are built and
> running; `python run_kerbside.py --scenario free_flow --open` starts the whole
> thing on one port. 289 tests pass, `ruff` is clean and `mypy --strict` is clean
> over `core/`, `geometry/` and `operators/`.
>
> **Three of the eight gates were not met, and none has been rewritten to fit.**
> Every gate was written before any code existed.
>
> | Phase | Gate | Outcome |
> |---|---|---|
> | 3 Detection | blob count matches truth in ≥ 98% of frames | **not met — 25%** |
> | 4 Measurement | counting ≥ 99%, speed MAE ≤ 2 km/h, lane ≥ 99% | **not met — 90.9% / 9.4 / 81%** |
> | 5 API | annotated stream ≥ 15 fps at 720p | **not met — 1.1 fps** |
> | 6 Console | calibrate through the UI to within 2% of the known homography | **met — 0.54% worst span** |
>
> Phases 1, 2, 6, 7 and 8 met theirs. The measured figures and the diagnosis are in
> [Detection accuracy](#detection-accuracy-phase-3-target-not-met) and
> [Measurement accuracy](#measurement-accuracy-phase-4) below. The Phase 3 and
> Phase 4 shortfalls are the same defect seen one stage apart; the Phase 5 one is
> separate and simpler.
>
> Everything claimed here is backed by a test in `tests/`. Nothing is claimed for
> a stage that does not exist, and nothing that falls short is reported as
> though it did not.

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

### The console

Build the interface once, then run the whole application from one command:

```bash
cd frontend && npm install && npm run build
```

```bash
.venv/Scripts/python.exe run_kerbside.py --scenario free_flow --open
```

FastAPI serves the REST API, the MJPEG stream, the telemetry WebSocket and the
built frontend from the same origin, so there is no CORS to configure and no
second server to start. `run_kerbside.py` puts `backend/` on the path, so the
console runs from a clean checkout without installing the package; with the
package installed, `python -m kerbside.serve` is equivalent.

If the frontend has not been built, the root route says so and explains how,
rather than returning a 404 that looks like a broken install. The REST API and
its OpenAPI documentation at `/docs` are available either way.

The console has nine views: the live **Console**, **Calibration**, the
**Pipeline** inspector, an **Operators** reference covering all 35 registered
operators by syllabus unit, the **Shadow** and **Motion** labs, **Analytics**,
the **Ablation** lab and the **Report**.

### Online, with no server

The same console also runs as a static website, which is how it is hosted on
Vercel. There is no server behind it: the **real Python pipeline runs inside the
visitor's browser** under [Pyodide](https://pyodide.org), CPython compiled to
WebAssembly, which ships NumPy, SciPy, OpenCV, Pydantic and FastAPI. Nothing is
ported or reimplemented in JavaScript; the `kerbside` package is bundled as it
is and imported in a Web Worker.

```bash
cd frontend && npm run build:web
```

That bundles `backend/kerbside` into the engine, copies `docs/` alongside, and
builds the frontend in its `web` mode into `frontend/dist-web/` - a separate
folder, so a web build can never replace what the local server serves. The frontend then hands
each request to `kerbside.api.inprocess.dispatch` instead of `fetch`, which
calls **the same FastAPI route functions** the server does, resolved from
FastAPI's own route table and validated by the same Pydantic models.
`tests/test_inprocess.py` holds it to answering exactly as the HTTP server does.

What differs, and why:

| | Local server | Browser build |
|---|---|---|
| Where the pipeline runs | a Python process | the visitor's device, in WebAssembly |
| Default clip | 720p, 12 s | 560×315, 5 s |
| First load | instant | about 35 MB, cached after; ~45 s to a ready console on a laptop |
| Video | MJPEG stream | the page steps the cursor and draws one frame at a time |
| Telemetry | WebSocket push | polled from `GET /api/telemetry` |
| Real video files | yes | no - there is no filesystem to read from |

The browser runs the pipeline about **4.5× slower** than native. The clip size
was chosen by measurement rather than guessed: at 480×270 for four seconds the
pipeline counted **no vehicles at all**, because they are too small at the
counting line to be confirmed as tracks; 560×315 for five seconds counts five
and analyses in under a minute, which beat 640×360 on both counts and time.

**On an iPad.** The console is laid out for tablets as well as laptops: the
inspector docks from 1024 px (every iPad in landscape) and opens from a
**Vehicles** button when held upright; controls grow to finger size; the
explanations a mouse user reads from hover tooltips are shown as text; and
"Add to Home Screen" opens it full-screen. `docs/demo-guide.html` - linked from
the overview at `/docs/` - is a step-by-step guide to preparing an iPad and
presenting the live console.

**Deploying:** import the repository into Vercel with the Root Directory left at
the repository root. `vercel.json` sets the install and build commands and the
output directory, so no project settings need changing.

`docs/kerbside-interface.html` is a single self-contained page showing what the
console renders, built from a real run. `docs/kerbside-presentation.html` is the
project presentation and `docs/kerbside-implementation-plan.html` the technical
plan; all three work offline on any browser.

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
| Calibrating **through the console API**, half-pixel clicks at 720p | 0.14% median span error, 0.54% worst — Phase 6 criterion met |
| Same, whole-pixel clicks | 0.27% median, 1.08% worst |
| Same, half-pixel clicks on a 320×180 frame | 2.05% — the criterion is resolution-dependent, because click error is what bounds it |

### Which background model, measured

The background study is the one place a hand-written model is scored against a
standard implementation. Free-flow clip, 8 s, 18–80 m working range:

| Model | Recall | Precision | Frame agreement | Count accuracy | fps |
|---|---|---|---|---|---|
| Running average (default) | 0.830 | 0.824 | 0.250 | 0.778 | 1.35 |
| Temporal median | 0.772 | 0.908 | **0.367** | **0.889** | 0.77 |
| Three-frame difference | 0.157 | 1.000 | 0.017 | 0.222 | 3.62 |
| First-frame reference | 0.774 | 0.760 | 0.222 | 1.000 | 1.39 |
| MOG2 (labelled baseline) | **0.951** | **0.934** | 0.294 | 0.444 | 1.22 |

Two things here are worth stating plainly. **MOG2 has the best recall and
precision and is not the best model for this job**: it over-counts badly, 14
vehicles against 9 expected, because its per-pixel mixture fragments a vehicle
more readily than a median over time does. Recall and precision are per-vehicle
measures and the gate is a count, so the model that wins on the first two loses
on the one that was set in advance.

And **the temporal median beats the default on both stated criteria** — frame
agreement 0.367 against 0.250, count accuracy 0.889 against 0.778 — at roughly
half the throughput and slightly lower recall. That is a real result on one
scenario and not yet a reason to change a default; changing it on one 8-second
clip would be fitting the default to the clip. It is recorded here as the study
output it is.

### The constraints are tested, not just stated

"No machine learning anywhere in the pipeline" is a property of every line in
`backend/kerbside/`, and "same video and parameters give identical output" is a
property of a run rather than of any function. Neither is the kind of claim a
reader can verify by looking at one file, so `tests/test_constraints.py` parses
every module in the package and asserts that nothing imports a learning
framework and nothing calls a pretrained detector, then runs the whole pipeline
twice and requires the two results to be identical down to each track's speed.

The single permitted exception is OpenCV's mixture-of-Gaussians background
subtractor, which the comparison study needs so the hand-written models can be
*measured* against a standard implementation rather than merely asserted to be
competent. It is pinned by test to one file and one clearly-labelled option, and
it is never the default.

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

## The pipeline

Six stages, each with its syllabus units, a rationale surfaced in the inspector,
and a validated parameter schema.

| Stage | Does | Units |
|---|---|---|
| S1 Ingest | Colour conversion, and the frame interval every speed divides by — measured from timestamps, not taken from the container | 1, 2 |
| S2 Stabilise | Phase-correlation shake compensation, with a confidence test that declines rather than guessing | 3 |
| S3 Background | Running average, temporal median, three-frame difference and a fixed reference; selective update; colour differencing | 2, 4 |
| S4 Foreground | Otsu, median filter, perspective-scaled opening and closing, hole filling | 2, 4, 6 |
| S5 Shadow | HSV shadow suppression with circular hue and a saturation-gated hue test | 7 |
| S6 Blobs | Connected components at 4/8/m connectivity, perspective size filter, shape plausibility, watershed splitting, lane assignment | 1, 6 |

### Detection accuracy (Phase 3 target NOT met)

Measured on the free-flow clip over an 18–80 m working range, vehicles of at
least 300 visible pixels, background warm-up excluded:

| Measure | Result | Target |
|---|---|---|
| Frame count agreement | **25.0%** | ≥ 98% |
| Recall | 0.830 | — |
| Precision | 0.823 | — |
| Mean absolute count error | 1.62 vehicles/frame | — |
| Merges / splits | 40 / 87 | — |
| Throughput at 720p | **4.0 fps** | ≥ 15 fps |

The limiting failure is **fragmentation**, not shadow and not detection: a
vehicle whose dark window band survives thresholding arrives as two components,
and counting them separately fails a count comparison exactly as a missed
vehicle does.

Two things were done about it and a third is still needed. The closing element's
height was being scaled by the wrong axis (see below), which alone accounted for
most of it. Then S6 gained a geometric fragment merge: every component's ground
point implies a place on the road, the homography says how tall and wide a
vehicle standing there would appear, and components that fit inside one such
silhouette are one vehicle in pieces — two genuine vehicles do not fit inside
one silhouette, which is what makes them two. That halved split events, from 144
to 87, and lifted recall from 0.80 to 0.83.

What remains is that frame *agreement* is an exacting measure — every frame must
carry precisely the right count — and at a mean absolute error of 1.6 vehicles
per frame it is not close. Reaching 98% needs recall and precision both above
0.99, which is roughly an order of magnitude better than the current 0.83. The
honest reading is that the mask is good enough to track from and not yet good
enough to count from frame by frame, and that closing the remaining gap is a
matter of the threshold and morphology behaving consistently across the whole
working range rather than of any one further insight.

Shadow suppression itself works: measured against shadow ground truth (two
renders identical but for `cast_shadows`), it removes **87–97% of true shadow
pixels** per frame. Turning it off does not improve the count, and rendering
with no shadows at all leaves the frame-agreement figure essentially unchanged —
which is how fragmentation was identified as the real constraint rather than
assumed away.

### Measurement accuracy (Phase 4 target NOT met)

Free-flow clip, 12 s, scored against exact per-frame ground truth. The gate asked
for counting ≥ 99%, speed MAE ≤ 2 km/h and lane ≥ 99%. **None of the three is
met.** What the two speed routes do deliver is a diagnosis of why.

| Measure | Result |
|---|---|
| Count accuracy | **90.9%** (12 counted, 11 expected) |
| Lane assignment | 81% correct |
| Size class | 52% correct |
| Speed, Route A — ground point | MAE **19.5 km/h**, bias −14.35 |
| Speed, Route B — block matching | MAE **9.4 km/h**, bias −3.88 |

Both speed routes are computed for every vehicle, because the comparison is the
finding rather than a diagnostic. They are built to fail in *independent* ways:

- **Route A** differences the bounding box's ground point through the road
  homography. That point is the only part of the silhouette genuinely on the
  calibrated plane. It carries a known bias — it maps to the vehicle's near
  extremity, half a length short — but a constant bias cancels under
  differencing, which is why an unbiased speed can be built on a biased
  position.
- **Route B** tiles the vehicle box, discards blocks with too little texture to
  say anything, matches the rest against the previous frame, takes the *median*
  displacement, refines it to sub-pixel by phase correlation, and maps it
  through the homography of the plane *at the height the blocks sit at*. Using
  the road-plane homography instead attributes body-panel motion to the ground
  and leaves an error that scales with that height, so it never averages away.

The blob centroid is the obvious third option and is rejected on geometry: a
centroid sits in the middle of the visible silhouette, which is not on the road
plane at all, and its error grows with distance and with vehicle height. It is
computed, and used for association only — never for speed.

**Route B wins by 2.1×, and the reason is consistent with Phase 3.** Route A's
position is defined by the *edge of the mask*, and that mask under-covers the
vehicle rear. A short mask shortens the measured displacement, which under-reads
the speed; the −14.35 km/h bias is that, measured. Route B never touches the
mask boundary. Two methods built to fail differently pointing at the same defect
is the strongest evidence in the project that segmentation, not measurement, is
the binding constraint.

Route B's residual is structural rather than a tuning problem: a sweep over the
smoothing window and gain moved its MAE between 9.4 and 9.6 km/h — essentially
flat. The remaining −3.9 km/h is not reachable from S8; it has to be fixed at
S4–S6, where the mask is made. The flat sweep is the result. Tuning until a
number looked better would have hidden it.

Class accuracy at 52% has the same cause. Footprint in metres is a sound basis
for classification and a fragmenting mask halves the footprint, so the class is
wrong for exactly the vehicles the mask splits.

### Throughput

| Pipeline | Rate at 720p | Target |
|---|---|---|
| Detection only, S1–S6 | **4.0 fps** | ≥ 15 fps |
| Full pipeline, S1–S9 with block matching | **1.1 fps** | ≥ 15 fps |

S8 dominates the full pipeline at roughly 570–640 ms per frame, because block
matching searches every textured block in every vehicle box. Both figures are
below target and both are reported, rather than quoting the faster one.

### Findings worth stating plainly

**A greyscale background subtraction cannot see a red vehicle.** BT.601 weights
red at 0.299, so on this project's palette against its own asphalt, a red body
differs from the road by **1.2 grey levels** — and by 94 in its red channel.
Blue, brown and green are within 10 levels too: four of ten body colours are not
merely hard to detect but invisible. The mask contains their shadow and not
them, which looks like a shadow problem and is a colour problem introduced two
stages earlier. The difference is therefore taken across colour channels.

**Hue is undefined where saturation is near zero, and asphalt is neutral.** At a
background saturation of about 0.075, the hue difference across foreground
pixels has a median of 10 and a 95th percentile of 67 — that is not a
measurement of colour, it is the arctangent of two small noisy differences.
Applying the textbook 10-unit hue tolerance there rejects half of a correctly
identified shadow, so the hue condition is enforced only above a saturation
floor.

**A closing element's height must scale with vehicle height, not with distance
along the road.** A silhouette extends upward from its ground point, and a metre
of height projects at close to the across-road scale — about 25 px/m at image
row 300 — while a metre along the road projects at about 4 px/m. Sized the wrong
way the element comes out two pixels tall, the dark window band across a vehicle
survives as a gap, and every vehicle is reported as two blobs. It presents as a
tracking problem and is a structuring element that was six times too short.

**Filling holes after removing a shadow restores the shadow.** Subtracting a
shadow from inside a larger foreground region leaves an *enclosed* hole, so an
unbounded hole fill puts the whole thing straight back. Bounding the fill by
area separates the two cases cleanly, because they differ by orders of
magnitude.

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
  stages/
    base.py              stage scaffolding: params, timing, frame context
    s1_ingest.py         decode, measure the true frame interval, convert
    s2_stabilise.py      phase-correlation shake compensation
    s3_background.py     running average, temporal median, three-frame
    s4_foreground.py     Otsu, median, perspective-scaled morphology
    s5_shadow.py         HSV shadow suppression
    s6_blobs.py          components, perspective size filter, splitting
    pipeline.py          orchestration, per-stage timings
    s7_track.py          world-frame prediction, association, coasting
    s8_speed.py          two independent speed routes, reported side by side
    s9_classify.py       size class from footprint in metres, colour naming
  eval/
    metrics.py           detection and counting metrics against ground truth
    analytics.py         flow, headway, percentiles, the fundamental diagram
    ablation.py          nine studies, each answering a stated question
    report.py            one self-contained HTML file, figures inlined
  api/
    routes.py            REST, MJPEG stream, telemetry WebSocket
    session.py           in-memory session, analysis, bounded frame retention
    overlay.py           the annotated frame; layers toggled independently
  serve.py               FastAPI app; serves the built frontend on one origin
  data/
    renderer.py          synthetic scene renderer with exact ground truth
    scenarios.py         presets, each targeting a specific failure mode
    degrade.py           resolution, JPEG, blur, noise, low light
    make_clips.py        CLI: render clips and ground truth to disk
frontend/src/
  lib/                   typed API client, telemetry socket, UI state
  components/            primitives, viewport, parameter rail, inspector
  views/                 console, calibration, pipeline, shadow, motion,
                         analytics, ablation, report
docs/                    presentation, implementation plan, interface tour
run_kerbside.py          start the console without installing the package
tests/                   pytest; every claim above is asserted here
```

## Syllabus mapping

Every one of the 35 registered operators records its syllabus unit, and the
console surfaces that unit next to the control that sets it — so the mapping is
a property of the running system rather than a table written afterwards.

| Unit | Where it carries weight |
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
