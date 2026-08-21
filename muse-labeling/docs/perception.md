# MUSE Labeling Tool

A browser-based tool for labeling radar–camera fusion data. Reconstructs the
four plots from `process.py` (RD map, radar detections, track history, camera)
and lets you label radar points as **object / noise / pair**, with each label
applying per-frame.

## Build

```bash
npm install
npm run build
```

## Docker Deploy

```bash
docker build -t muse-labeling .
docker run -p 443:443 muse-labeling
```

Open Chrome and go to `https://localhost` — click "Advanced" → "Proceed" to
bypass the self-signed certificate warning. (HTTPS/localhost is required for the
File System Access API used to read and write the data folder.)

## Data Format (`yolo_tracking_data.json`)

Produced by `process.py`. Each top-level key is a frame; everything needed to
reconstruct that frame's plots and store its labels lives under it — no frame
depends on another.

```json
{
  "frame_00100.jpeg": {
    "frame_index": 100,
    "source_camera_file": "01779276639.145422.jpeg",
    "rd_raw_file": "rd_raw/frame_00100_rd.raw",

    "box_detections": [
      {
        "pos1": [780.87, 503.90],
        "pos2": [1408.36, 954.52],
        "label": "car",
        "confidence": 0.866,
        "track_id": 1,
        "thickness": 2,
        "original": {
          "pos1": [780.87, 503.90],
          "pos2": [1408.36, 954.52],
          "label": "car",
          "confidence": 0.866
        }
      }
    ],

    "radar_detections": [
      {
        "track_id": 0,
        "is_confirmed": true,
        "centroid": { "range_m": 13.5, "velocity_kmh": -18.2 },
        "points": [
          { "range_m": 13.2, "velocity_kmh": -17.8 }
        ]
      }
    ],

    "track_history": [
      {
        "track_id": 0,
        "history": [
          { "range_m": 15.2, "velocity_kmh": -18.0 },
          { "range_m": 13.5, "velocity_kmh": -18.2 }
        ]
      }
    ],

    "labeling": {
      "object": { "1": 7 },
      "noise":  [9],
      "pairs":  [[12, 15]]
    }
  }
}
```

### Per-frame fields

| Field | Plot | Description |
|-------|------|-------------|
| `rd_raw_file`      | RD map (1st)  | Path to the raw `rd_power` matrix (`.raw`). The frontend reads it and renders the RD map itself — see below |
| `box_detections`   | Camera (4th)  | YOLO + ByteTrack boxes. `pos1` top-left, `pos2` bottom-right (pixel coords). `track_id` is the candidate key within the frame. `original` keeps the untouched YOLO output for restore |
| `radar_detections` | Radar (2nd)   | Radar clusters after CFAR + DBSCAN + tracking. `track_id` is both the per-frame candidate key and the persistent track identity |
| `track_history`    | Track (3rd)   | Full trajectory of each confirmed track up to this frame — self-contained, no need to read other frames |
| `labeling`         | —             | Labels for this frame. Filled by this tool; `process.py` emits it empty |

### RD map rendering (`.raw`)

The RD map is **not** stored as a rendered image. `process.py` writes the raw
`rd_power` matrix to `rd_raw/frame_*_rd.raw` (float32, little-endian, row-major,
256×256), and the frontend renders it on a canvas, reproducing `process.py`'s
visualization exactly:

```
10 * log10(rd_power)  →  transpose (.T)  →  clip to [0, 30]  →  gray_r (invert)  →  origin="lower"
```

This keeps the full-precision matrix (nothing is downsampled) and lets the
frontend control colormap / scaling without re-running `process.py`.

### `box_detections.original` — YOLO restore point

Each box carries an `original` snapshot of the YOLO output as it was first
produced. Editing `pos1` / `pos2` / `label` in the tool never touches it, so the
box can always be reset back to the detector's result via **Restore original
YOLO box**. Manually drawn boxes have no `original` (nothing to restore to).

### `labeling` — per-frame single source of truth

Each frame owns its labeling. This is deliberate: the radar tracker can make
mistakes (ID switches across frames), so labels are never assumed to carry over.

| Field | Structure | Meaning |
|-------|-----------|---------|
| `object` | `{ "radarId": boxId \| null }` | This radar track is a real object. Value is the YOLO box id it maps to **this frame**, or `null` if the camera didn't detect it |
| `noise`  | `[radarId, ...]` | Radar tracks that are noise |
| `pairs`  | `[[a, b], ...]` | Mirror/ghost pairs. The symmetric relation is stored **once** per pair |

**Rules enforced by the tool:**
- `object` and `noise` are mutually exclusive per radar id.
- Marking one half of a pair as `noise` cascades to its partner; un-marking does too.
- A point can be both paired and (object **or** noise) at once — the marker shows nested hollow outlines.

## Output folder layout

`process.py` writes one folder per run (`--frames-dir`):

```
my_output_frames/
├── yolo_tracking_data.json   # all per-frame data + labels
├── rd_raw/                   # raw rd_power matrices (.raw) — RD map source
├── rd/                       # rendered RD map jpegs (reference only; not used by the tool)
├── cluster/                  # rendered cluster plots (reference)
├── track/                    # rendered track plots (reference)
├── camera/                   # raw camera frames (used by the tool)
├── camera_yolo/              # camera frames with YOLO boxes drawn (reference)
└── combined/                 # 4-in-1 combined plots (reference)
```

The tool only needs `yolo_tracking_data.json`, `rd_raw/`, and `camera/`. The
other image folders are rendered references for eyeballing.

## Labeling marker legend

| Marker | Meaning |
|--------|---------|
| ✕ | radar centroid (always shown) |
| ○ green circle outline | object |
| △ gray triangle outline | noise |
| □ colored square outline | pair (color is unique per pair) |

## Usage

1. Click **Open Folder** and select your `my_output_frames/` directory.
2. **← →** to navigate frames; type a number in the top bar and Enter to jump.
3. **Camera:** drag to draw a new box, click to select, drag corners to resize,
   edit coordinates/type in the panel, **Del** to delete, **Restore original
   YOLO box** to undo edits back to the detector output.
4. **Radar:** click a centroid (✕) to select, then toggle Object / Pair / Noise
   in the panel.
   - **Object** asks you to click the matching camera box (or "object with no box").
   - **Pair** asks you to click the partner point.
   - **Esc** cancels a pending pair/box selection.
5. **Ctrl+S** to save · **Ctrl+Z / Ctrl+Shift+Z** to undo/redo.

## Batch pre-labeling (optional)

For simple scenes, `prelabel_object.py` pre-fills `labeling.object` before manual
review — it binds a radar track to a box across all frames:

<!-- ```bash
python prelabel_object.py yolo_tracking_data.json --radar-id 1 --box-id 1
```

Per frame: if radar 1 is absent it's skipped; if box 1 is absent the value is
`null`; otherwise `object["1"] = 1`. A `.bak` backup is written before overwriting. -->