# Build Prompt — Kerbside

Paste everything below the horizontal rule into Claude Code as your first message, in an empty directory.

---

You are building a complete, production-quality academic project from scratch. Read this entire brief before writing any code. Work through the phases in order and do not skip the verification gate at the end of each phase.

# 0. What you are building

**Kerbside** — a traffic analytics system that takes video from a **fixed roadside camera** and produces vehicle counts, per-vehicle speeds in km/h, lane assignment, vehicle class and colour, using **only classical digital image processing**. It ships with a polished operations console that lets a user watch every operator in the pipeline run, change its parameters live, and see accuracy change in real time.

This is a university course project for a subject called **Image and Video Processing**. It is graded on how well it demonstrates classical DIP techniques, so the interface is not decoration — it is the primary evidence that the techniques work and are understood.

## The core premise

A camera bolted to a pole sees a road. From nothing but the pixels and a one-time geometric calibration, recover: how many vehicles passed, in which lane, in which direction, how fast, what size class, what colour. No detector, no tracker-by-detection, no learned model anywhere.

This is what deployed traffic counting and speed enforcement systems did for two decades before deep learning, and what many still do. The classical pipeline is not a compromise here — it is the historical and still-practical solution.

# 1. Hard constraints — do not violate these

1. **No machine learning anywhere in the pipeline.** No neural networks, no YOLO, no Haar cascades, no HOG+SVM, no `cv2.dnn`, no pretrained weights, no scikit-learn classifiers. Vehicle *classification* is done by real-world footprint size derived from the homography, not by a trained model.
2. **Background modelling must be implemented by hand.** Running-average and temporal-median background models written with NumPy. `cv2.createBackgroundSubtractorMOG2` / `KNN` may be included **only** as a clearly-labelled reference baseline in the comparison study — never as the production path. The point of the project is that you built the model.
3. **Every operator must be explainable.** Each pipeline stage carries a short human-readable rationale and the syllabus unit it belongs to, surfaced in the UI. This is a course requirement.
4. **CPU only.** No CUDA. Target ≥ 15 fps at 720p on a laptop.
5. **Deterministic.** Same video and parameters produce identical output. Seed every RNG.
6. **No enforcement claims.** The system measures speed; it must never assert legal accuracy, issue a "violation", or imply evidentiary standing. Label over-limit rows as "flagged for review". Say plainly in the README that a legally admissible speed device requires type approval and calibration certification that this project does not have.
7. **No personal data.** Do not read, store or attempt to recognise number plates or faces. If you build any privacy feature, build blurring, not recognition.

# 2. The syllabus this must satisfy

Every unit must be genuinely load-bearing — not an operator bolted on to tick a box.

| Unit | Content | Where it must appear |
|---|---|---|
| 1 | Digital image representation, image as a matrix, spatial and gray-level resolution, distance measures (Euclidean, city-block, chessboard, m-path), file formats | Frame extraction and colour conversion; a resolution study (1080p / 720p / 480p / 360p — real CCTV is often poor); distance measures used as the gating metric for track association, compared |
| 2 | Point processing (negative, contrast stretching, thresholding, intensity slicing, log, power-law); neighbourhood processing (smoothing, sharpening, median, Gaussian); histogram equalisation; Fourier spectrum and frequency-domain filtering | **Temporal background modelling is a temporal low-pass filter** — make this framing explicit; median filtering of the foreground mask; contrast stretching and histogram equalisation for night and backlit frames; frequency-domain removal of periodic rain/interference patterns |
| 3 | DFT, Hadamard transform, DCT and its application to image compression | **Phase correlation (DFT-based) for sub-pixel displacement**, used both for camera-shake stabilisation and to refine speed; a DCT compression study answering a real question — how far can CCTV bitrate drop before counting degrades; Hadamard in the transform comparison |
| 4 | Dilation, erosion, opening, closing, hit-or-miss, boundary extraction, thinning, thickening, greyscale morphology | The core of foreground cleanup: opening to kill speckle, closing to fill vehicle interiors, **perspective-scaled structuring elements**; boundary extraction for vehicle outlines; greyscale morphology for illumination-gradient background estimation |
| 5 | First- and second-order derivative edge detection; Prewitt, Sobel, Laplacian, LoG, Canny | Lane-marking detection (Canny + Hough) to derive lane polygons and help calibration; edge-based refinement of blob boundaries under motion blur; a five-operator comparison on identical frames |
| 6 | Pixel relationships, component labelling (4-, 8-, m-connectivity), thresholding, Otsu, region growing, region splitting and merging | Otsu on the background-difference image; **connected-component labelling individuates vehicles**; a 4 vs 8 vs m connectivity study; watershed / region splitting to separate merged blobs |
| 7 | Colour fundamentals and models (RGB, CMY, CMYK, HSI, YUV, HSV), model transformation, colour slicing, colour histograms | **HSV shadow suppression — the headline result**; a colour-space comparison for shadow discrimination (RGB vs HSV vs YCbCr vs normalised rgb vs c1c2c3); vehicle colour naming from an HSV histogram of body pixels |
| 8 | Video representation, temporal correlation, I/P/B frame classification, motion estimation, exhaustive search, 2-D log search, three-step search | **Block-matching speed estimation as an alternative to centroid tracking**, with all three search algorithms compared; frame classification driving a motion-triggered recording mode, which is what real DVRs do |

# 3. Technology stack — use exactly this

**Backend**
- Python 3.11+
- FastAPI + Uvicorn
- NumPy, OpenCV (`opencv-python-headless`), SciPy
- Pydantic v2 for request/response models
- pytest
- `pyproject.toml`, installable with `pip install -e .`

**Frontend**
- React 18 + TypeScript + Vite
- Tailwind CSS v3
- Zustand for client state, TanStack Query for server state
- Recharts for charts
- `lucide-react` for icons
- `framer-motion`, used sparingly (see §5)

**Rules**
- The frontend builds to static assets **served by FastAPI** at the root path. `python -m kerbside.serve` starts everything with no Node required on the target machine.
- No opinionated component library (no MUI, Ant, Chakra). Build the components. Radix UI primitives only for the accessibility plumbing of sliders, dropdowns, dialogs and tooltips.
- No CSS-in-JS. Tailwind plus a `tokens.css` of CSS custom properties.
- **Transport:** annotated video streams to the browser as **MJPEG** over a plain HTTP endpoint; per-frame telemetry (tracks, counts, speeds, timings) goes over a **WebSocket**; single-frame pipeline-stage inspection and ablations are ordinary REST. Do not try to push raw frames over the WebSocket.

# 4. Repository layout

```
kerbside/
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── backend/
│   └── kerbside/
│       ├── serve.py                  # FastAPI app: REST + MJPEG + WebSocket + static
│       ├── api/
│       │   ├── routes_stream.py
│       │   ├── routes_pipeline.py
│       │   ├── routes_calibration.py
│       │   ├── routes_analytics.py
│       │   ├── routes_ablation.py
│       │   └── schemas.py
│       ├── core/
│       │   ├── types.py              # Blob, Track, Detection, Lane, Calibration, StageResult
│       │   ├── registry.py           # operator registry: id, name, unit, params, rationale
│       │   └── cache.py
│       ├── geometry/
│       │   ├── homography.py         # DLT from scratch + cv2 cross-check
│       │   ├── calibration.py        # 4-point road-plane calibration, scale bar, validation
│       │   └── perspective.py        # expected pixel size / SE size as a function of image row
│       ├── stages/                   # the pipeline, one module per stage
│       │   ├── s1_ingest.py          # decode, resolution, true fps measurement, colour convert
│       │   ├── s2_stabilise.py       # global motion estimation, camera-shake compensation
│       │   ├── s3_background.py      # running average, temporal median, three-frame difference
│       │   ├── s4_foreground.py      # difference, Otsu, morphology, hole filling
│       │   ├── s5_shadow.py          # HSV shadow suppression
│       │   ├── s6_blobs.py           # connected components, perspective size filter, splitting
│       │   ├── s7_track.py           # nearest-neighbour association + constant-velocity predictor
│       │   ├── s8_speed.py           # centroid route AND block-matching route
│       │   └── s9_classify.py        # footprint-based class, HSV colour naming
│       ├── operators/                # pure functions, no I/O
│       │   ├── point.py              # negative, contrast stretch, threshold, log, power-law, slicing
│       │   ├── spatial.py            # box, gaussian, median, laplacian, unsharp, high-boost
│       │   ├── frequency.py          # dft, spectrum, notch/low/high pass, phase correlation
│       │   ├── transforms.py         # dct, idct, hadamard, compression study
│       │   ├── morphology.py         # erode, dilate, open, close, hit-or-miss, boundary, thin
│       │   ├── edges.py              # prewitt, sobel, laplacian, log, canny
│       │   ├── segment.py            # otsu, region growing, watershed, components 4/8/m
│       │   ├── colour.py             # rgb/hsi/hsv/yuv/ycbcr/c1c2c3 conversions, slicing, histograms
│       │   └── motion.py             # SAD, exhaustive / 2-D log / three-step block matching
│       ├── data/
│       │   ├── renderer.py           # synthetic traffic scene renderer WITH exact ground truth
│       │   ├── scenarios.py          # free-flow, congested, night, rain, occlusion, shake
│       │   └── degrade.py            # resolution, JPEG, motion blur, noise, low light
│       └── eval/
│           ├── metrics.py
│           ├── ablation.py
│           └── report.py
├── frontend/
│   └── src/
│       ├── styles/tokens.css
│       ├── lib/{api.ts,ws.ts,store.ts,format.ts}
│       ├── components/
│       └── views/
│           ├── ConsoleView.tsx        # live/playback with overlays
│           ├── CalibrationView.tsx
│           ├── PipelineView.tsx
│           ├── ShadowLab.tsx
│           ├── MotionLab.tsx
│           ├── AnalyticsView.tsx
│           ├── AblationLab.tsx
│           └── ReportView.tsx
├── data/
│   ├── clips/                        # short sample clips
│   └── calib/                        # saved calibrations
└── tests/
```

# 5. Design system — follow this precisely

The look is **a traffic operations console**: dark, dense, quiet, legible at a glance from two metres away. Not a SaaS landing page, not a neon "AI surveillance" aesthetic, no glassmorphism, no scanline effects.

## Colour tokens

Define in `tokens.css`, map into `tailwind.config.ts` as named colours. Never write a raw hex in a component.

```
--bg-base:        #0B0F14
--bg-surface:     #121821
--bg-raised:      #1A2230
--bg-overlay:     #212B3A
--border-subtle:  #1E2733
--border-default: #2A3644
--border-strong:  #3A4959

--text-primary:   #E8EEF4
--text-secondary: #98A6B5
--text-muted:     #63717F
--text-inverse:   #0B0F14

--accent:         #3FA7D6   /* primary action, active state, selected track */
--accent-hover:   #55B9E4
--accent-dim:     #3FA7D626

--flow-free:      #2FBF71   /* free-flowing, within limit, improvement */
--flow-slow:      #E8B931   /* slow / congested / warning */
--flow-stop:      #E5484D   /* stopped / over limit / regression */
--shadow-hl:      #A78BFA   /* shadow-suppression module accent */
--motion-hl:      #F97362   /* motion-estimation module accent */
```

**Ship dark first and make it flawless.** A light theme is optional; if built, redefine the same token names under `[data-theme="light"]` — never conditional colours in components.

## Typography

- UI text: **Inter** (`@fontsource-variable/inter`), weights 400/500/600.
- **Every number** — speeds, counts, pixel coordinates, metres, timings, parameters, fps: **JetBrains Mono**, `font-variant-numeric: tabular-nums`. This single rule is most of what makes it read as an instrument.
- Bundle fonts locally. No CDN link.

| Role | Size / weight |
|---|---|
| View title | 20px / 600 |
| Panel header | 11px / 600 / 0.08em / uppercase / muted |
| Body | 13px / 400 |
| Label | 12px / 500 / secondary |
| Numeric readout | 13px mono / 500 / tabular |
| Big stat | 28px mono / 600 / tabular |
| Overlay label (on video) | 11px mono / 600, on a 70%-opacity chip so it stays legible over any frame |

## Spacing, shape, motion

- 4px scale. Panel padding 16px, gap 12px, control height 28–32px.
- Radius 6px controls, 8px panels, 10px dialogs. Nothing rounder.
- Borders over shadows. One shadow, for overlays only: `0 8px 24px rgba(0,0,0,.5)`.
- **Motion budget 120–180ms, ease-out, opacity and 2–4px transform only.** No springs, no bounce, no staggered lists, no page transitions. Exceptions: the metric-delta flash (§6.4) and the count-increment tick.
- Focus ring on every interactive element: 2px `--accent` at 40%, offset 2px.

## Component rules

- **Panels** = uppercase header left, controls right, hairline divider, content. Consistent everywhere.
- **Sliders** always show their numeric value in mono to the right of the label.
- Never centre body text. Numbers right-aligned in tables.
- **No decorative accent bars, gradient headers, or edge stripes on cards.** Separate with background tint and spacing.
- Empty states carry one line of instruction plus the control that resolves them.
- Icons 16px, lucide-react, `--text-secondary`. No emoji in the UI.

## Video overlay rules

The overlay is where this project is judged, so treat it as a designed surface:

- Bounding boxes: 1.5px, colour by state — tracked (`--accent`), newly detected (white, fading to accent over 200ms), counted (`--flow-free` flash), flagged over-limit (`--flow-stop`).
- Draw the **ground point** as a small filled circle at the bottom-centre of each box — this is the point that is actually on the road plane and it must be visible, because it is the thing the speed is computed from.
- Track history as a fading polyline behind each vehicle, 12 frames.
- Counting line per lane, dashed, with a live count chip at its end.
- Lane polygons at 8% fill opacity so they never fight the video.
- Every overlay layer independently toggleable, with the toggles in one panel and their state persisted.

# 6. Interface specification

## 6.1 Shell

Three regions, each scrolling independently.

- **Left rail (216px, collapsible to 48px).** The pipeline as a vertical spine: S1 Ingest → S2 Stabilise → S3 Background → S4 Foreground → S5 Shadow → S6 Blobs → S7 Track → S8 Speed → S9 Classify. Each row shows name, syllabus unit chips, per-stage ms, and a status dot. Below a divider: Calibration, Shadow Lab, Motion Lab, Analytics, Ablation Lab, Report.
- **Centre.** The active view.
- **Right rail (300px).** Inspector — parameters for the selected stage, its rationale, and its syllabus unit.
- **Bottom console (collapsible, 160px).** Live log: timestamp, stage id, parameters that changed, wall-clock ms, resulting metric delta. Monospace, auto-scroll, pause on hover. Build this properly — it is what makes the app feel like instrumentation.

Persistent status bar: source name, resolution, **measured** fps vs container fps, processing fps, dropped-frame count, calibration state (calibrated / uncalibrated — and speeds must be visibly disabled when uncalibrated).

## 6.2 Console view (primary screen)

**Source bar**: pick a synthetic scenario (free-flow / congested / night / rain / heavy-occlusion / camera-shake, with sliders for vehicle density, speed range, shadow strength) or load a video file, or use a bundled clip. Show format, resolution, bit depth, duration, container fps and measured fps in mono — that is Unit 1 information and it belongs on screen.

**Main canvas**: the MJPEG stream with overlays. Pan and zoom (0.5×–8×) with a fit button. Play / pause / step-one-frame / scrub, and a speed control (0.25×–4×). Clicking a vehicle selects it and pins its detail card.

**Timeline** under the video: a density strip showing per-frame motion energy with **I/P frame classification colour-coded**, plus tick marks where counting events occurred. Scrubbable.

**Live panel (right of video, or docked below)**: total count, count per lane, per class, current mean speed, 85th-percentile speed (the standard traffic engineering statistic), flow rate in vehicles/hour, and a rolling speed histogram.

**Selected vehicle card**: track id, lane, class with the footprint dimensions in metres that produced it, colour name with its HSV swatch, instantaneous and mean speed from both the centroid route and the block-matching route side by side, track age, and a sparkline of its speed over the track's life.

## 6.3 Calibration view

Speed is meaningless without geometry, so make this view genuinely good.

- Pause on a clear frame. The user clicks **four points on the road plane** forming a quadrilateral — ideally the corners of a rectangle whose real dimensions they know (lane edges and two lateral markings).
- They enter the real-world width and length in metres. Compute the homography.
- Show, side by side: the original frame with the quad and a projected **1-metre grid** overlaid on the road surface, and the rectified bird's-eye view. If the metre grid does not look square and evenly spaced on the road in the original, the calibration is wrong — say so in the UI.
- Provide a **validation tool**: click two points, get the real-world distance between them. The user checks it against a tape measure or a known marking.
- Show the reprojection residual in pixels and a plain-language quality verdict.
- Lane definition: draw lane polygons by clicking, or auto-suggest them from Canny + Hough lane-marking detection and let the user adjust.
- Save and load calibrations as JSON keyed by camera/clip.
- **Assist, don't assume:** offer common reference distances (lane width, dashed-marking pitch) as *suggestions the user must verify against their own road* — never as hard-coded truth. Local standards vary; a tape measure is the ground truth.

## 6.4 Pipeline view and live recompute

A stage strip of thumbnails, one per stage, each labelled with operator and unit chip. Selecting one shows its full-resolution output on the main canvas with a **swipe-compare divider** against any other stage, plus a **pixel probe** on hover reporting coordinate, RGB, HSV, background value, difference value, and whether the pixel passes the current threshold and why.

Changing a parameter re-runs from that stage forward on the current frame and updates the metrics. Make it feel instant:

- Debounce 120ms while dragging, run immediately on release.
- Cache each stage keyed by a hash of (frame id + all upstream parameters).
- Work at preview resolution (long edge ≤ 900px) during a drag, reconcile at full resolution on release.
- **Metric delta flash**: animate changed numbers with a ± badge for 1.5s, `--flow-free` for improvement, `--flow-stop` for regression.
- "Reset to default" per parameter, and a badge on the panel when anything differs from default.

Add an **"apply to whole clip"** button that re-runs the current parameters over the full video and updates the aggregate accuracy against ground truth. Single-frame feedback is for exploration; clip-level numbers are what get reported.

## 6.5 Shadow Lab — the headline result

Dedicate a view to it, because this is the most compelling thing in the project.

- Four synchronised panels: raw frame, foreground mask **without** shadow suppression, the HSV decision maps (value ratio, saturation difference, hue difference — three small heatmaps with their thresholds drawn on them), and the foreground mask **with** suppression.
- Below: the connected-component view for both, side by side, with the component count printed on each.
- **The money shot**: a jump-to-frame control that finds frames where suppression changed the count, and a case list showing "2 vehicles merged into 1 blob by a shadow bridge → split correctly after suppression". Precompute this list over the clip.
- Live sliders for the four thresholds, with the count delta over the whole clip updating on release.
- A colour-space comparison tab: run shadow discrimination in RGB, HSV, YCbCr, normalised rgb and c1c2c3 on identical frames and report separability and resulting count error for each.

## 6.6 Motion Lab (Unit 8)

- Frame pair selector with the reference and current frame side by side.
- **Motion vector field** as arrows per 16×16 macroblock over the current frame, coloured by error against ground truth where known, with a toggle to show only textured blocks.
- Click any block to see its **SAD search surface** as a 3-D-shaded heatmap with the paths taken by exhaustive, 2-D log and three-step search drawn on it. This makes it visually obvious why fast searches fall into local minima — build it, it is the best teaching artefact in the project.
- Algorithm comparison: MV error, exact-vector %, motion-compensated prediction PSNR, block-comparison count, wall-clock ms, speed-up.
- **Speed route comparison**: for every tracked vehicle, centroid-derived speed vs block-matching-derived speed vs ground truth, as a scatter and a residual plot. State which route wins and under what conditions.
- Motion-triggered recording panel: frame classification over the clip, percentage of frames that would be stored, and the implied storage saving — the real-world application of Unit 8.

## 6.7 Analytics view

The operations dashboard a traffic engineer would actually read.

- Cumulative and per-interval counts, split by lane, direction and vehicle class.
- Speed distribution histogram with mean, median and **85th percentile** marked.
- Speed vs time, and flow (veh/hr) vs time.
- A fundamental-diagram scatter: flow against density, which is the classic traffic-engineering plot and shows you understand the domain.
- Headway distribution (time gap between successive vehicles).
- A table of vehicles flagged over the configured limit — labelled "flagged for review", never "violation", with the disclaimer from §1.6 visible.
- Export everything to CSV.

## 6.8 Ablation Lab

Pick an axis and a metric, sweep, get a chart plus a table. Ship these presets one click away:

1. **Shadow suppression on/off** — count error, false-merge rate, speed MAE.
2. **Background model** — running average (α sweep) vs temporal median (window sweep) vs three-frame difference, against count accuracy and ghosting incidence.
3. **Morphological SE size and shape** — fixed vs perspective-scaled, against false merges and missed distant vehicles.
4. **Connectivity** — 4 vs 8 vs m for blob labelling, with disagreement pixels highlighted.
5. **Input resolution and compression** — 1080p/720p/480p/360p × JPEG quality, against count and speed accuracy. This answers a real question about CCTV bitrate.
6. **Block-matching algorithm** — exhaustive vs 2-D log vs three-step, accuracy against comparison count.
7. **Edge operator** — Prewitt, Sobel, Laplacian, LoG, Canny for lane-marking detection.
8. **Colour space for shadow discrimination** — RGB vs HSV vs YCbCr vs normalised rgb vs c1c2c3.
9. **Scenario robustness** — the full pipeline across free-flow / congested / night / rain / occlusion / shake.

Every result exportable to CSV and PNG.

## 6.9 Report view

One button generates a self-contained HTML and PDF report with every study, table and figure, plus the exact parameters and the calibration used.

# 7. Algorithm specification

## S1 — Ingest
- Decode with OpenCV. Record resolution, channel count, bit depth, container fps, codec.
- **Measure the true frame interval** from presentation timestamps rather than trusting container metadata — every speed in the system is divided by this number, so a wrong fps produces confidently wrong speeds. Report both and warn when they disagree by more than 2%.
- Greyscale by ITU-R BT.601 (0.299R + 0.587G + 0.114B), weights exposed as parameters.

## S2 — Stabilise
- Pole-mounted cameras vibrate. Estimate global frame-to-frame translation by **phase correlation on the DFT** of a static region of the frame (or the full frame), and warp to compensate.
- Report the residual shake amplitude in pixels. Provide a toggle so the user can see what happens to the foreground mask without stabilisation — the answer is that the whole frame lights up, which is a good demonstration.

## S3 — Background modelling
Implement three, all by hand:
- **Running average**: `B ← (1−α)B + αI`, α exposed. Frame this explicitly as a temporal low-pass filter (Unit 2).
- **Temporal median** over a sliding window of N frames — more robust to stopped vehicles, more memory.
- **Three-frame difference** — no background at all, good for shake, poor for slow vehicles.
- **Selective update**: freeze background updating inside pixels currently labelled foreground, to stop stopped vehicles being absorbed. Make this a toggle and show the ghosting it prevents.
- Maintain the background in **both greyscale and HSV**, because S5 needs the HSV background.

## S4 — Foreground extraction
- Absolute difference against the background; **Otsu** to threshold (compare against a fixed threshold and against a per-pixel adaptive threshold).
- Median filter to remove salt-and-pepper.
- Morphological **opening** with a small SE to remove speckle, then **closing** with a larger SE to fill vehicle interiors (windscreens and windows often match the background more closely than the body does).
- **Perspective-scaled structuring elements**: SE size is a function of image row, derived from the homography — see trap 4.
- Hole filling by morphological reconstruction or flood fill from the border.

## S5 — Shadow suppression (the headline)
For each foreground pixel, compare against the HSV background. Classify as shadow when all three hold:
- **Value ratio** `V_frame / V_bg ∈ [α, β]` with `0 < α < β < 1` — darker, but not black (a black car is not a shadow). Start near α ≈ 0.4, β ≈ 0.85.
- **Saturation difference** `S_frame − S_bg ≤ τ_S` — shadows desaturate slightly. Start near τ_S ≈ 0.1 of full scale.
- **Hue difference** `|H_frame − H_bg| ≤ τ_H`, computed **with circular wraparound** — shadows largely preserve hue. Start near τ_H ≈ 10 in OpenCV's 0–179 scale.

All four thresholds are user parameters with live histograms showing where they sit relative to the actual distributions. Removing shadow pixels from the mask is what splits vehicles that a shadow had bridged into one blob.

Implement the same test in the alternative colour spaces for the §6.8 comparison.

## S6 — Blobs
- Connected components with 8-connectivity by default; 4 and m available for the study.
- **Perspective-aware size filter**: for each candidate blob, project a reference vehicle footprint (e.g. 4.0 m × 1.8 m) onto the image at that blob's road position via the homography, and accept the blob if its pixel area falls within a tolerance band of the expected area. A single global min-area threshold cannot work — see trap 3.
- **Splitting merged blobs**: when a blob's real-world footprint is implausibly large, attempt a split using the distance-transform watershed, or by vertical-projection valley finding along the lane direction. Report split events; they are interesting.
- Compute for each blob: bounding box, **ground point (bottom-centre of the box)**, area, real-world footprint, lane assignment by point-in-polygon on the ground point.

## S7 — Tracking
- Greedy nearest-neighbour association between the previous frame's tracks and this frame's blobs, gated by a maximum association distance **measured in metres on the road plane, not pixels** — a pixel gate is wrong under perspective.
- Constant-velocity prediction to place the search centre; a Kalman filter is acceptable (it is linear estimation, not learning) and is a good optional upgrade.
- Track lifecycle: tentative → confirmed after k hits → coasting for up to m missed frames → terminated.
- Make the distance metric a parameter (Euclidean / city-block / chessboard) so Unit 1's distance measures are exercised and compared for real.
- **Counting**: a track is counted when its ground point crosses its lane's counting line in the expected direction. Debounce so a track hovering on the line cannot double-count.

## S8 — Speed, two independent routes
**Route A — track-based.** Convert the ground point to road-plane coordinates via the homography for two frames separated by Δt; speed = Δdistance / Δt. Smooth over a short window. Report in km/h.

**Route B — block-matching (Unit 8).** Take the vehicle's bounding box in the current frame, tile it into 16×16 blocks, discard blocks whose intensity variance is below a texture threshold, and run block matching against the previous frame with a ±16 px search range and SAD cost. Take the **median** motion vector across surviving blocks as the vehicle's image displacement, refine to sub-pixel by **phase correlation**, then map through the homography.

Implement **exhaustive search, 2-D logarithmic search and three-step search**, all three, and compare on motion-vector accuracy, exact-vector percentage, motion-compensated prediction PSNR, and block-comparison count.

Then compare Route A against Route B against ground truth. Route B should prove more stable when the blob boundary is unreliable — partial occlusion, shadow at the edge, motion blur — because it measures texture displacement rather than blob shape. State the finding either way; a negative result honestly reported is fine.

Also classify frames I or P by frame-difference energy and drive the motion-triggered recording statistic.

## S9 — Classification and colour
- **Class by real-world footprint**, not appearance: convert the blob's ground-plane extent to metres and bucket into two-wheeler / car / LCV / truck-or-bus using documented length and width bands, with aspect ratio as a tie-breaker. Write the bands down as named constants in the code — the report must state them, because otherwise the classification is unfalsifiable.
- **Colour naming**: HSV histogram over body pixels with the darkest and brightest deciles excluded (windows and specular highlights), take the dominant hue bin, map to a named colour through a documented table, and handle achromatic cases (low saturation → white/grey/black by value).

# 8. Data and ground truth

**Build the synthetic renderer first and build it carefully — it is the foundation of every number in the project.**

`data/renderer.py` renders a fixed-camera road scene by projecting simple vehicle sprites through a known homography onto a road plane, and records, per frame, for every vehicle: id, world position in metres, lane, speed in km/h, image bounding box, ground point, class, colour, and an occlusion flag. That is exact ground truth for counting, speed, lane assignment and classification, with zero hand annotation.

Must support:
- Configurable camera height, tilt and focal length, so perspective is real and the homography is known exactly.
- Multiple lanes, both directions, lane markings at a configurable real-world pitch.
- Vehicles of different footprints and colours, entering at configurable rates and speeds, including **stop-and-go** behaviour.
- **Cast shadows** rendered as an offset, sheared copy of the vehicle darkened in *value* while preserving hue and saturation — this is what makes the S5 result demonstrable and it must be physically sensible.
- Occlusion: vehicles in the far lane partly hidden by vehicles in the near lane.
- Scenario presets: free-flow, congested, night (low light, headlight glare), rain (streaks plus reduced contrast), heavy occlusion, camera shake (sinusoidal jitter of known amplitude).

`data/degrade.py` applies resolution reduction, JPEG recompression, motion blur proportional to speed, sensor noise, and low-light gamma.

**Real footage tier.** Ship instructions, not footage: how to shoot a validation clip — camera fixed and stable, oblique view of the road, two ground marks a **tape-measured** distance apart in frame, a note of the recording fps. Provide a stopwatch-based cross-check workflow: time a vehicle between the two marks by hand and compare against the system. Report synthetic and real results **separately and never conflated**.

Do not download large datasets or require any account, API key or licence.

# 9. Known traps — handle these up front

1. **Speed from the blob centroid is systematically wrong.** The centroid sits above the road plane, so it does not obey the road-plane homography. Use the **ground point** — bottom-centre of the bounding box — for every geometric conversion. Getting this wrong produces speeds that look plausible but carry a consistent bias that grows with vehicle height, so trucks read differently from cars and it never looks like a bug.
2. **Cast shadows bridge adjacent vehicles into one blob**, so the count is silently low and it presents as a tracking failure rather than a colour-processing failure. Fix it in S5 before you debug S7.
3. **A single minimum-area threshold cannot work under perspective.** Set it high enough to reject noise near the camera and distant vehicles vanish; set it low and near-field noise becomes vehicles. Scale the threshold by expected projected area at that image row.
4. **Closing with too large a structuring element merges vehicles in adjacent lanes**; too small leaves vehicle interiors hollow and one car becomes three blobs. Scale the SE with perspective too, and keep it anisotropic — vehicles project wider than tall near the horizon.
5. **Ghosting in the running-average background.** A vehicle stopped at a signal gets absorbed into the background, then leaves a vehicle-shaped hole when it moves off, which is detected as a phantom. Use selective update or the temporal median.
6. **Hue wraps around at 0/179 in OpenCV.** A naive `abs(H1 − H2)` says red and red are maximally different. Use circular difference in the shadow test or it will fail on exactly the vehicles that matter.
7. **Container fps often lies.** Measure the real frame interval from timestamps. Every speed divides by it.
8. **Camera shake makes the entire frame foreground.** Stabilise before background subtraction, or your first night of testing on real footage will look like total failure.
9. **The aperture problem in block matching.** A plain-sided white truck has no texture; blocks inside it match everywhere equally. Gate motion-vector scoring by block variance and take the median rather than the mean across blocks.
10. **Tie-break block matching toward zero displacement**, or uniform regions pick whichever minimum the loop saw first and the vector field becomes noise.
11. **Motion blur biases the bounding box** in the direction of travel, which shifts the ground point and inflates speed. Note it, measure it against the synthetic ground truth, and prefer Route B where blur is heavy.
12. **Night is a different parameter regime, not a broken pipeline.** Headlights saturate the value channel and cast moving light pools that behave like objects. Ship a separate night parameter preset and say so, rather than tuning one set of thresholds to be mediocre everywhere.

# 10. Build phases

Do not start a phase until the previous gate passes. Commit at each gate.

**Phase 1 — Renderer, geometry and ground truth.** `data/renderer.py`, `data/scenarios.py`, `data/degrade.py`, `geometry/homography.py`, `geometry/calibration.py`, `core/types.py`.
*Gate:* render a 720p free-flow clip with known per-vehicle speeds and shadows; dump ground truth to JSON; write a test asserting that projecting a vehicle's known world position through the known homography lands on its recorded image ground point to within 1 px; and that DLT-recovered homography matches the rendering homography.

**Phase 2 — Operator library.** Everything under `operators/`, pure functions, plus `core/registry.py` giving each operator an id, name, syllabus unit, parameter schema and one-sentence rationale.
*Gate:* unit tests for every operator including known-answer tests — a 3×3 average filter on a hand-worked matrix, Otsu on a synthetic bimodal histogram, a DFT round-trip, block matching on a synthetically shifted patch recovering the exact shift.

**Phase 3 — Stages S1–S6 and metrics.** Ingest, stabilise, background, foreground, shadow, blobs. `eval/metrics.py`.
*Gate:* on the Phase 1 free-flow clip, per-frame blob count matches ground-truth vehicle count in ≥ 98% of frames, and enabling shadow suppression measurably reduces false merges. If it does not, the bug is trap 2, 3 or 4.

**Phase 4 — Stages S7–S9.** Tracking, counting, both speed routes, classification, colour.
*Gate:* counting accuracy ≥ 99% and speed MAE ≤ 2 km/h on the free-flow synthetic clip; lane assignment ≥ 99%; both speed routes implemented and cross-compared.

**Phase 5 — Backend API.** REST, MJPEG stream, WebSocket telemetry, stage cache, preview downsampling, calibration persistence.
*Gate:* the annotated stream runs at ≥ 15 fps at 720p, telemetry arrives without backpressure stalls, and single-frame stage inspection returns in under 400 ms warm.

**Phase 6 — Frontend shell, Console and Calibration views.** Tokens, component primitives, three-region shell, video canvas with overlays, timeline, live panel, inspector with live recompute, bottom console, full calibration workflow with the metre grid and the distance-validation tool.
*Gate:* calibrate a synthetic clip through the UI and land within 2% of the known homography; the projected metre grid looks square on the road; dragging a parameter visibly changes the mask and flashes the metric delta. Screenshot it and check against §5 before moving on.

**Phase 7 — Shadow Lab, Motion Lab, Analytics.** Including the SAD search-surface visualisation and the speed-route comparison.
*Gate:* the shadow case list finds real merge-split events and jumping to one shows the count change; exhaustive search reaches < 1 px MV error on synthetic frames while costing at least 20× more block comparisons than 2-D log search.

**Phase 8 — Ablation Lab, report, docs, polish.** All nine ablation presets, `eval/report.py`, README with screenshots, `CLAUDE.md`, one-command run, accessibility and visual pass.
*Gate:* `python -m kerbside.serve` works on a clean checkout; the report button produces a complete PDF; every interactive element is keyboard-reachable with a visible focus ring.

# 11. Acceptance criteria

Synthetic targets are **hard gates** — the renderer gives exact ground truth, so there is no excuse for missing them. Real-footage targets are to be **characterised honestly**, not guaranteed.

| Study | Target |
|---|---|
| Count accuracy, free-flow synthetic | ≥ 99% |
| Count accuracy, heavy-occlusion synthetic | ≥ 92%, with the failure modes catalogued |
| Speed MAE, free-flow synthetic, exact calibration | ≤ 2 km/h |
| Speed MAE, congested synthetic | ≤ 4 km/h |
| Lane assignment accuracy | ≥ 99% |
| Class accuracy (4 buckets, footprint-based) | ≥ 90% |
| Shadow suppression benefit | ≥ 50% relative reduction in false-merge events; report the count-error delta |
| Homography recovery from 4 clicked points | within 2% of the known rendering homography |
| Block matching, exhaustive, synthetic | MV MAE < 1 px, > 90% exact vectors |
| Fast searches vs exhaustive | ≥ 20× fewer block comparisons, with measurably worse accuracy |
| Throughput | ≥ 15 fps end-to-end at 720p on a laptop CPU |
| Resolution degradation | characterise count and speed accuracy at 1080p/720p/480p/360p — degradation is expected and must be reported, not hidden |
| Real footage, hand-calibrated | speed within ±10% of a tape-measure-and-stopwatch reference. State plainly that certified enforcement devices achieve far tighter tolerances with type approval this project does not have |

# 12. Engineering standards

- Type hints everywhere; `mypy --strict` clean on `core/`, `operators/`, `geometry/`.
- TypeScript `strict: true`; no `any`.
- Every operator is a pure function taking arrays and explicit parameters. All I/O lives in the API layer.
- Docstrings on operators state the syllabus unit, the mathematical definition, and parameter meanings.
- **Every physical quantity carries its unit in the variable name** — `speed_kmph`, `dist_m`, `dx_px`, `dt_s`. Mixing pixels and metres silently is the most likely source of a wrong answer in this project.
- pytest coverage ≥ 80% on `operators/` and `stages/`.
- Handle errors visibly: a video that will not decode, a missing calibration, a parameter set producing zero detections — each gets a specific message in the UI, never a silent empty state or a stack trace.
- `README.md` documents install, run, the stage map, the syllabus mapping table, the calibration procedure, every study with its result, and the §1.6 limitation statement.

# 13. What not to do

- Do not add a database. In-memory plus files on disk.
- Do not add authentication or multi-tenancy.
- Do not add Docker until everything else is finished.
- Do not read, store or recognise number plates or faces.
- Do not call anything a "violation", or imply legal or evidentiary accuracy.
- Do not write marketing copy. Labels are nouns; descriptions state what an operator does.
- Do not silently swallow exceptions.
- Do not reach for a neural network when a stage is struggling. Fix the stage.

---

Start with Phase 1. Before writing code, restate the Phase 1 plan in a few lines and confirm the renderer's ground-truth contract — exactly what is recorded per vehicle per frame, how the camera geometry is parameterised, and how a test will verify that the recorded image ground point matches the projection of the recorded world position.
