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
   SubtractorMOG2`/`KNN` may appear *only* as a clearly labelled reference
   baseline in the comparison study, never on the production path.
3. **Every operator carries a rationale and its syllabus unit**, surfaced in the
   UI. This is a course requirement, not a nicety.
4. **CPU only.** Target ≥ 15 fps at 720p on a laptop.
5. **Deterministic.** Same video and parameters give identical output. Seed
   every RNG.
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
  api/        FastAPI routes; all I/O lives here
frontend/src/ React + TypeScript + Vite, served as static assets by FastAPI
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

## Commands

```bash
.venv/Scripts/python.exe -m pytest -q                        # test suite
.venv/Scripts/python.exe -m kerbside.data.make_clips --list  # scenarios
.venv/Scripts/python.exe -m kerbside.data.make_clips --all   # regenerate clips
.venv/Scripts/python.exe -m ruff check backend tests
.venv/Scripts/python.exe -m mypy backend/kerbside/core backend/kerbside/operators backend/kerbside/geometry
```

A project virtualenv lives in `.venv` because the system Python has a
NumPy/SciPy version mismatch. Use `.venv/Scripts/python.exe` on Windows.

## Build phases

Do not start a phase until the previous gate passes. Commit at each gate.

1. **Renderer, geometry, ground truth** — done.
2. Operator library (`operators/`) with known-answer tests.
3. Stages S1–S6 and `eval/metrics.py`.
4. Stages S7–S9: tracking, counting, both speed routes, classification.
5. Backend API: REST, MJPEG, WebSocket telemetry, stage cache.
6. Frontend shell, Console and Calibration views.
7. Shadow Lab, Motion Lab, Analytics.
8. Ablation Lab, report, docs, polish.

## What not to do

- No database. In-memory plus files on disk.
- No authentication, no multi-tenancy.
- No Docker until everything else is finished.
- No opinionated component library on the frontend; Radix primitives only for
  accessibility plumbing.
- Never call anything a "violation" or imply legal accuracy.
- Never reach for a neural network when a stage is struggling. Fix the stage.
