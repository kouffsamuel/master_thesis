# Download Dataset

https://drive.google.com/drive/u/0/folders/1Qr1kLIdO9L7VWHLWhmpd9lcJZvnpPadE

# Other Soruces

https://drive.google.com/drive/u/1/folders/1XvDidBrudjbi7LmMIHW9hRLKcudPm60R

# MUSE Labeling Tool

A browser-based tool for labeling radar–camera fusion data. Reconstructs the
four plots from `process.py` (RD map, radar detections, track history, camera)
and lets you label radar points as **object / noise / pair**, with each label
applying per-frame.

## Build

```bash
npm install
npm run build
npm run dev
```

For local labeling, keep `npm run dev` running and open the Vite URL it prints
(usually `http://localhost:5173`). `npm run build` only creates the production
files in `dist/`; it does not start the labeling tool.

## Opening Data

1. Click **Open Folder**.
2. Select the exported data folder, for example:

   ```text
   /Users/aurora/Desktop/my_output_frames_day_1
   ```

3. If the folder contains multiple labeling JSON files, use the top-bar JSON
   dropdown to choose which one to open.

The selected folder must contain the image/radar assets used by the JSON:

```text
yolo_tracking_data.json
camera/
rd_raw/
```

For split/chunk JSON files, keep them in the same folder as `camera/` and
`rd_raw/`. The tool reuses those folders; it does not need copied images.

## Large JSON Files

Chrome should not load multi-GB labeling JSON files in one shot. The frontend
will refuse to auto-load a JSON file larger than 400MB and will ask you to pick
a smaller split JSON instead.

If `yolo_tracking_data.json` is very large, split it into smaller JSON chunks in
the same exported data folder:

```bash
python split_label_json.py /Users/aurora/Desktop/my_output_frames_day_1/yolo_tracking_data.json \
  --frames-per-file 500
```

This writes files such as:

```text
yolo_tracking_data_00000_00499.json
yolo_tracking_data_00500_00999.json
```

For `my_output_frames_day_1`, the current chunk set is:

```text
yolo_tracking_data_00000_00499.json
yolo_tracking_data_00500_00999.json
yolo_tracking_data_01000_01499.json
yolo_tracking_data_01500_01999.json
yolo_tracking_data_02000_02499.json
yolo_tracking_data_02500_02999.json
yolo_tracking_data_03000_03499.json
yolo_tracking_data_03500_03999.json
yolo_tracking_data_04000_04147.json
```

After opening the folder, choose one chunk from the dropdown. **Save** writes
back to the currently selected JSON chunk, not to the original multi-GB
`yolo_tracking_data.json`. If there are unsaved changes and you switch chunks,
the tool asks for confirmation first.

### File access troubleshooting

If Chrome shows:

```text
The requested file could not be read, typically due to permission problems that have occurred after a reference to a file was acquired.
```

re-click **Open Folder** and select the exported data folder again. This usually
means Chrome had a folder/file handle, but macOS or the browser could no longer
read the underlying files. Common fixes:

- Use Chrome or Edge on `localhost` / HTTPS; Safari and Firefox do not fully
  support the File System Access API used by this tool.
- Select the run folder that directly contains `yolo_tracking_data.json`,
  `camera/`, and `rd_raw/`.
- Do not move, rename, delete, or cloud-sync the selected folder while the tool
  is loading it.
- On macOS, allow Chrome access to Desktop/Documents/Downloads if the data lives
  there: **System Settings → Privacy & Security → Files and Folders**. If that
  still fails, grant Chrome **Full Disk Access** or move the data folder to a
  non-protected local directory.
- If the page was refreshed or reopened, select the folder again; browser file
  handles are not reliable across sessions for this workflow.

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

### Relaxed radar threshold export

For labeling, the radar export can be generated with a more relaxed detector
than the original visualization path in `process.py`.

Original `process.py` settings:

```python
CA_CFAR(win_param=(15, 20, 9, 10), threshold=12, rd_size=(N, N))
DBSCAN(eps=2, min_samples=3)
```

Relaxed labeling export settings in `MUSE/Processing/export_label_json_relaxed.py`:

```bash
python MUSE/Processing/export_label_json_relaxed.py \
  --output-dir ./label_export_relaxed_cfar9_dbscan2 \
  --cfar-threshold 9.0 \
  --dbscan-eps 2.0 \
  --dbscan-min-samples 2 \
  --peak-metric mean
```

What changed:

| Parameter | Original | Relaxed labeling export | Effect |
|-----------|----------|-------------------------|--------|
| `cfar_threshold` | `12` | `9.0` | Lower CFAR threshold keeps weaker radar peaks, so fewer possible targets are missed before manual review |
| `dbscan_eps` | `2` | `2.0` | Same neighborhood radius as the original path |
| `dbscan_min_samples` | `3` | `2` | Allows smaller clusters to survive DBSCAN instead of being dropped as noise |
| `peak_metric` | implicit processing choice | `mean` | Stores cluster power using the mean peak metric for the relaxed export |

Each exported frame records these values under `metadata`, for example:

```json
"metadata": {
  "cfar_threshold": 9.0,
  "dbscan_eps": 2.0,
  "dbscan_min_samples": 2,
  "peak_metric": "mean"
}
```

The relaxed export is intentionally recall-heavy: it may include more radar
noise, but the labeling tool can mark those candidates as noise or pairs later.
If the export is run with `--skip-yolo`, `box_detections` will be empty; restore
YOLO boxes from an earlier `ori_yolo_tracking_data.json` by copying
`box_detections` frame-by-frame before pre-labeling object mappings.

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
2. If a JSON dropdown appears in the top bar, choose the chunk/file to label.
3. **← →** to navigate frames (**Shift+← →** jumps 10 frames); type a number in
   the top bar and Enter to jump.
4. **Camera:** drag to draw a new box, click to select, drag corners to resize,
   edit coordinates/type in the panel, **Del** to delete, **Restore original
   YOLO box** to undo edits back to the detector output.
5. **Radar:** click a centroid (✕) to select, then toggle Object / Pair / Noise
   in the panel.
   - **Object** asks you to click the matching camera box (or "object with no box").
   - **Pair** asks you to click the partner point.
   - **Esc** cancels a pending pair/box selection.
6. **Ctrl+S** to save · **Ctrl+Z / Ctrl+Shift+Z** to undo/redo.

## Batch pre-labeling (optional)

For simple scenes, `prelabel_object.py` pre-fills `labeling.object` before manual
review — it binds a radar track to a box across all frames:

<!-- ```bash
python prelabel_object.py yolo_tracking_data.json --radar-id 1 --box-id 1
```

Per frame: if radar 1 is absent it's skipped; if box 1 is absent the value is
`null`; otherwise `object["1"] = 1`. A `.bak` backup is written before overwriting. -->
