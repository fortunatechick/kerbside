# Kerbside — working notes for Claude Code

Traffic analytics from a fixed roadside camera using **only classical digital
image processing**. University course project for Image and Video Processing;
the interface is the primary evidence that the techniques work, not decoration.

## Non-negotiable constraints

1. **No machine learning anywhere in the pipeline.** No neural networks, no
   YOLO, no Haar cascades, no HOG+SVM, no `cv2.dnn`, no pretrained weights, no
   scikit-learn classifiers. Vehicle classification is by real-world footprint
   from the homography, never by appearance.
2. **Background modelling is written by hand** with NumPy. `cv2.createBackground
   SubtractorMOG2` appears *only* as the `mog2_baseline` model in S3, which is a
   clearly labelled reference the comparison study measures the hand-written
   models against; it is never the default and never on the production path.
   KNN does not appear at all. `tests/test_constraints.py` pins both facts -
   the baseline to one file and one option, and its description to wording a
   reader actually sees.
3. **Every operator carries a rationale and its syllabus unit**, surfaced in the
   UI. This is a course requirement, not a nicety.
4. **CPU only.** Target ≥ 15 fps at 720p on a laptop.
5. **Deterministic.** Same video and parameters give identical output. Seed
   every RNG. Asserted for the renderer *and* for the pipeline: two runs must
   agree on blob counts, stage metrics and every per-track speed, lane and
   class, and re-seeding NumPy's global generator between runs must change
   nothing - if it does, something is drawing from the global RNG and the run
   is reproducible only until someone else draws a number first.
6. **No enforcement claims.** The system measures speed. Over-limit rows are
   "flagged for review", never "violations", and the README states plainly that
   a legally admissible speed device needs type approval this project lacks.
7. **No personal data.** Never read, store or recognise plates or faces. Any
   privacy feature is blurring, not recognition.

## Layout

```
backend/kerbside/
  core/       types, operator registry, cache
  geometry/   homography (from-scratch DLT), calibration, perspective scaling
  operators/  pure functions: point, spatial, frequency, transforms,
              morphology, edges, segment, colour, motion
  stages/     s1_ingest .. s9_classify, the pipeline proper
  data/       synthetic renderer with exact ground truth, scenarios, degradation
  eval/       metrics, ablation, report
  api/        FastAPI routes, session, overlay; all I/O lives here
  serve.py    the app; serves the built frontend from the same origin
frontend/src/ React + TypeScript + Vite, served as static assets by FastAPI
  lib/        typed API client, telemetry socket, UI state
  components/ primitives, viewport, parameter rail, inspector
  views/      console, calibration, pipeline, shadow, motion, analytics,
              ablation, report
docs/         presentation, implementation plan, interface tour (standalone HTML)
tests/        pytest
data/clips, data/gt, data/calib   build output, not committed
```

## Conventions that are load-bearing

- **Units in every identifier.** `speed_kmph`, `dist_m`, `dx_px`, `dt_s`,
  `elapsed_ms`. Mixing pixels and metres silently is the most likely source of a
  wrong answer in this project.
- **World frame:** +X across the road, +Y along it away from the camera, +Z up.
  Road surface is the plane Z = 0.
- **Image frame:** x right, y down, origin top-left.
- **Homographies** are stored normalised so `h[2, 2] == 1`.
- **Operators are pure functions** taking arrays and explicit parameters. All
  I/O lives in the API layer.
- Docstrings on operators state the syllabus unit, the mathematical definition
  and what the parameters mean.
- Handle errors visibly. A video that will not decode, a missing calibration, a
  parameter set producing zero detections: each gets a specific message. Never a
  silent empty state, never a swallowed exception.

## Facts established by measurement, not assumed

These came out of Phase 1 and the tests assert them. They are here because
getting any of them wrong later would produce plausible-looking wrong numbers.

- **The bounding-box ground point maps to the vehicle's near extremity**, not to
  its centre — exactly half a vehicle length short, and exactly so for an
  unrolled camera, because the image row of a road point depends only on
  distance along the road. Absolute position therefore carries a `length / 2`
  bias, but it is constant per vehicle, so it *cancels under differencing*.
  That is why S8 can build an unbiased speed on a biased position.
- **The centroid is a far worse ground estimate**, and its error grows with both
  distance and vehicle height (correlation ≈ 0.85 with height on the free-flow
  clip). That is trap 1: it never looks like a bug, it looks like trucks being
  slower than cars.
- **Shadow illumination is multiplicative, not additive.** A shadowed surface is
  still lit, by the sky rather than the sun, which multiplies reflectance per
  channel. Modelling the ambient as an additive glow inflates the saturation of
  near-neutral asphalt to about +0.19 of full scale, far outside anything S5's
  threshold would accept and far outside what a camera records.
- **Calibration accuracy is limited by click precision, not by the estimator.**
  Exact clicks recover the rendering homography to 1e-9. Half-pixel clicking
  stays inside 2% over the 18–60 m tracked road; whole-pixel clicking is ~1.2%
  in the median with a tail past 2%. Hence: offer zoom, and prefer a large
  calibration quadrilateral.
- **Container fps can be made to lie** on purpose via `write_video(...,
  declared_fps=...)`, so S1's timestamp measurement is a tested behaviour rather
  than a claim.

From Phase 2, the operator library:

- **m-connectivity and 8-connectivity always induce identical components.** If a
  shared 4-neighbour of a diagonal pair is foreground, a 4-path already joins
  them; if neither is, m admits the diagonal directly. Proven, and asserted over
  random masks. The connectivity ablation must therefore report count
  differences for 4 vs 8, and *path ambiguity* for m — claiming a count
  difference for m would be reporting something that cannot happen.
- **Otsu's criterion is flat across an empty histogram gap.** Every level inside
  the gap separates the modes identically, so a plain `argmax` returns the
  lowest — hard against the upper tail of the dark mode. On a difference image
  that admits far more noise than the middle of the gap. `otsu_threshold` takes
  the plateau midpoint by default; `plateau_midpoint=False` reproduces the
  textbook and OpenCV answer exactly.
- **A unimodal Gaussian scores 0.64 separability**, because splitting one mode
  at its median puts the halves ~1.6σ apart. Any "is it bimodal" cut below that
  is meaningless; the line is at 0.85.
- **Phase correlation needs upsampled-DFT refinement, not a parabolic fit.** For
  a fractional shift the inverse transform of a pure phase ramp is a periodic
  sinc with alternating side lobes, so a three-point parabola lands ~1 px out —
  OpenCV's `phaseCorrelate` has the same problem. With refinement: < 0.1 px.
- **Motion vectors are reported as content displacement, not codec convention.**
  Positive `dx_px` means the texture moved right. S8 adds a block-match vector,
  a phase-correlation refinement and a centroid velocity together; three
  quantities that get added must not disagree about which way is positive.
  `BlockMatch.reference_offset_px` gives the codec form.
- **Spectral peak detection has to be scale-invariant.** The prominence of
  ordinary image content grows with image size — measured at 12, 29 and 63 on
  128 px, 256 px and 720p road frames — purely because a larger annulus offers
  more bins for the maximum. Genuine interference scores over 1000 against a
  runner-up of 10, so the test is dominance over the *same image's* ceiling.
- **A notch filter must pin the DC term to unity.** A Gaussian notch of radius 6
  placed 12 px out leaves DC at 0.87, and two take it to 0.76; losing a quarter
  of the image mean dwarfs the interference removed.
- **In image coordinates a +45° gradient points down-right.** Rows increase
  downward, so non-maximum suppression for that sector compares (+1,+1) and
  (−1,−1). Getting the diagonal sectors the wrong way round silently deletes
  diagonal edges — it cost 0.86 → 0.40 IoU against OpenCV before it was found.

From Phase 3, the stages:

- **Greyscale background subtraction cannot see a red vehicle.** BT.601 weights
  red at 0.299, so on this project's own palette against its own asphalt, red is
  within **1.2 grey levels** of the road, and blue, brown and green within 10 —
  four of ten body colours. Their *channel* difference is 30 to 94. The default
  difference is therefore the max across channels; `difference_mode="grey"` is
  kept only as the baseline that demonstrates the problem.
- **Hue is undefined where saturation is near zero, and asphalt is neutral**
  (saturation ≈ 0.075). Enforcing a 10-unit hue tolerance there rejects half of
  a correctly-identified shadow, because the "hue" being compared is the
  arctangent of two small noisy differences. S5 applies the hue condition only
  above a saturation floor.
- **A closing element's *height* must be scaled by vehicle height, not by
  distance along the road.** A silhouette extends upward from its ground point,
  and a metre of height projects at close to the across-road scale (≈25 px/m at
  row 300) while a metre along the road projects at ≈4 px/m. Sized the wrong
  way the element is a couple of pixels tall, the dark window band survives as a
  gap, and **every vehicle reports as two blobs** — a roof and a bumper.
- **Filling holes after shadow removal restores the shadow.** Subtracting a
  shadow from inside a larger foreground region leaves an *enclosed* hole, so an
  unbounded `fill_holes` puts it straight back. The restoration is bounded by
  area: a windscreen is hundreds of pixels, a cast shadow tens of thousands.
- **Selective update needs the previous frame's mask handed to it.** The frame
  context is rebuilt per frame, so `ctx.foreground` is always `None` when S3
  runs; keying the update on it meant the feature silently never fired.
- **The stabiliser aligns to the first frame**, so what it reports is
  displacement *relative* to that frame. The first frame's own offset survives
  as a constant — harmless, because the background model is built in the same
  frame of reference, but it must be subtracted before comparing against
  absolute ground truth.

And two mistakes in the *evaluation*, which were costing the pipeline more than
its actual defects:

- **The visibility gate says what must be found, not what may be found.** Gating
  ground truth by range while scoring every blob counts a correct detection just
  outside the range as a false positive. Worst exactly where it matters least: a
  one-pixel ground-point error at 100 m is about eight metres.
- **Recall counts vehicles found, not blob-vehicle pairs.** A vehicle reported
  as two fragments has been found once. Crediting it twice let recall exceed
  1.0, which is how the error surfaced.

From Phases 4 and 5, measurement and the API:

- **The two speed routes are the ground point and block matching, not the
  centroid.** The field names say so: `speed_kmph_track` and
  `speed_kmph_blockmatch`. They were once named for a centroid the stage no
  longer uses, which is a naming trap of exactly the kind this file exists to
  record — a reader would have believed Route A was the centroid route the whole
  design argues *against*. The blob centroid is still computed, and is used for
  association only.
- **Block matching beats the ground point by 2.1×, and the reason is the mask.**
  Route A's position is the edge of the mask, and that mask under-covers the
  vehicle rear; a short mask shortens the displacement and under-reads the
  speed, which is the −14.35 km/h bias measured. Route B never touches the mask
  boundary. Two routes built to fail independently agreeing on one defect is
  the strongest evidence in the project that segmentation is the constraint.
- **Route B must use the homography of the plane at the blocks' height.** Body
  texture sits about a metre above the road; mapping its displacement through
  the *road* homography attributes it to the wrong plane and leaves an error
  that scales with that height, so it does not average away.
- **Route B's residual is not tunable.** A sweep over the smoothing window and
  gain moved its MAE between 9.4 and 9.6 km/h. Fixing it means fixing S4–S6.
- **A FrameContext holds ~35 MB of float32 intermediates at 720p.** Retaining
  one per frame cost gigabytes over a clip and died partway through with an
  `ArrayMemoryError`; because the failure left `_analysis_done` False, the next
  request restarted the analysis and the console never recovered. Contexts now
  release their intermediates beyond `Session.RETAINED_CONTEXTS` frames, and
  `Session.context(intermediates=True)` replays the clip when the stage
  inspector needs detail that has gone. `OverlayOptions.needs_intermediates`
  declares which layers care, so an ordinary scrub never triggers a replay and
  the MJPEG stream never does.
- **A replay rewinds the tracker's running totals**, so the totals are frozen
  when an analysis completes and `summary()` reports the frozen ones.
- **FastAPI cannot build a response model from a union of `Response`
  subclasses.** The SPA fallback route annotated `FileResponse | HTMLResponse`
  and raised at import time — but only once `frontend/dist` existed, which is
  why an earlier API smoke test passed. It carries `response_model=None`.
- **S8 dominates the frame budget** at roughly 570–640 ms, so the full S1–S9
  pipeline runs at 1.1 fps against the 4.0 fps of detection alone. Both numbers
  are reported; quoting only the faster one would describe a pipeline that does
  not measure anything.

- **An unknown `stage_id` escaped as a 500.** `set_params` read the stage's
  current parameters *before* the `try`, so the `KeyError` that names the nine
  real stages never reached the handler that turns it into a 400. Anything that
  can raise belongs inside the guard, not next to it.
- **`/api/operators` is populated by import side effect.** `@register` runs when
  the operator module is imported, so code that reaches `REGISTRY` without
  importing `kerbside.operators` sees an empty registry and no error. The count
  is pinned at 35 by a test.

- **MOG2 wins on recall and precision and loses on the gate.** In the background
  study it scores 0.951 / 0.934 per vehicle against the temporal median's
  0.772 / 0.908, and then counts 14 vehicles where 9 exist. Its per-pixel
  mixture fragments a vehicle more readily than a median over time, and the
  gate is a count. Do not read a per-vehicle metric as if it were the gate.
- **The temporal median beats the running-average default** on frame agreement
  (0.367 vs 0.250) and count accuracy (0.889 vs 0.778), at half the throughput.
  Measured on one 8-second free-flow clip, which is why the default has not
  moved: changing it on one clip would be fitting the default to the clip. A
  2.5-second clip put the same figure at 0.809, which is the reminder that a
  short clip is a different experiment and not a cheap version of a long one.

## Performance notes

The pipeline went from 1.28 fps to 4.0 fps at 720p during Phase 3. What mattered:

- `np.minimum.at` / `np.maximum.at` over a flattened label image is ~23 ms per
  call on 720p and was called three times a frame. `scipy.ndimage.find_objects`
  does the same work in one C pass.
- `fill_holes` by iterated geodesic reconstruction converges only after as many
  dilations as the longest path inside a hole — hundreds of full-frame passes.
  A flood fill from outside is one traversal.
- Extracting a component as `labels == label` over the whole frame costs frame
  area × component count. Slice the bounding box first.
- float32 throughout the mask stages: the input is 8-bit and nothing here needs
  15 significant figures.

## Commands

```bash
.venv/Scripts/python.exe -m pytest -q                        # 280 tests
.venv/Scripts/python.exe run_kerbside.py --scenario free_flow --open
.venv/Scripts/python.exe -m kerbside.data.make_clips --list  # scenarios
.venv/Scripts/python.exe -m kerbside.data.make_clips --all   # regenerate clips
.venv/Scripts/python.exe -m ruff check backend tests
.venv/Scripts/python.exe -m mypy backend/kerbside/core backend/kerbside/geometry backend/kerbside/operators --strict
cd frontend && npm run build                                 # then reload the console
```

`run_kerbside.py` puts `backend/` on `sys.path` and hands over to
`kerbside.serve`, so the console starts without installing the package — which
matters on a machine where the interpreter's `Scripts/` directory is not
writable and `pip install -e .` fails partway.

A project virtualenv lives in `.venv` because the system Python has a
NumPy/SciPy version mismatch. Use `.venv/Scripts/python.exe` on Windows.

## Build phases

Do not start a phase until the previous gate passes. Commit at each gate.

**Gates are written before the code and never rewritten afterwards.** A gate
chosen after the measurement is not a gate, it is a description. Three of these
eight are not met; that is recorded, not adjusted. If a phase's gate is not
measured, the status is "not measured" and never "passed".

1. **Renderer, geometry, ground truth** — done.
2. **Operator library** (`operators/`) with known-answer tests — done.
3. **Stages S1–S6 and `eval/metrics.py`** — built and tested, **gate not met**.
   Frame-count agreement is 25% against a 98% target and throughput is 4.0 fps
   against 15. The limiting failure is fragmentation (144 split events against
   38 merges, halved to 87 by the geometric merge): a vehicle reported as two
   blobs fails a count comparison just as surely as a missed one. Enlarging the
   closing element trades fragmentation for vehicle-to-vehicle merging rather
   than fixing either, which is why S6 separates the two by *shape* — a fragment
   and its sibling stack vertically within one plausible silhouette, whereas two
   vehicles do not. **The gate stays recorded as failed.** It was set before any
   code was written; lowering it afterwards would make it meaningless.
4. **Stages S7–S9** — built, **gate not met**. The gate was counting ≥ 99%,
   speed MAE ≤ 2 km/h, lane ≥ 99%, both routes cross-compared. Measured: count
   90.9%, lane 81%, class 52%, Route A (ground point) 19.5 km/h MAE, Route B
   (block matching) 9.4. The cross-comparison is done and is the useful part;
   the three accuracy criteria are not met, and the count and class shortfalls
   are the Phase 3 mask seen one stage later.
5. **Backend API** — built, **throughput gate not met**. REST, MJPEG, WebSocket
   telemetry, stage cache, 34 end-to-end tests. The gate asked for an annotated
   stream at ≥ 15 fps at 720p; the full pipeline runs at 1.1 fps because S8
   block matching costs ~600 ms a frame.
6. **Frontend shell, Console and Calibration** — built, **gate met**.
   Calibrating through `/api/calibration` with half-pixel clicks on a 720p frame
   lands at 0.14% median span error and 0.54% worst, against a 2% criterion;
   whole-pixel clicks give 0.27% / 1.08%. On a 320x180 frame the same half-pixel
   jitter gives 2.05% and misses — **the criterion is resolution-dependent**,
   because click error and not the estimator is what bounds it. Measure this at
   the resolution the user actually clicks on.
7. **Shadow Lab, Motion Lab, Analytics** — done, gate met on the motion half:
   exhaustive search at 0.32 px vector error, 32–36× the cost of the fast
   searches against a ≥ 20× criterion.
8. **Ablation Lab, report, docs** — done. `tests/test_api.py` exercises every
   endpoint end to end, which is what Phase 5's gate actually asks for; driving
   the routes by hand was not it, and three of the defects found while wiring
   the console would have been caught by it. 280 tests pass; ruff clean; mypy
   strict clean over `core/`, `geometry/`, `operators/`.

## What not to do

- No database. In-memory plus files on disk.
- No authentication, no multi-tenancy.
- No Docker until everything else is finished.
- No opinionated component library on the frontend; Radix primitives only for
  accessibility plumbing.
- Never call anything a "violation" or imply legal accuracy.
- Never reach for a neural network when a stage is struggling. Fix the stage.
