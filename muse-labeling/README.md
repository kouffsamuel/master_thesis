# OpenHub MUSE Label Tool

A lightweight browser-based image annotation tool for drawing bounding boxes on images from the OpenHub MUSE project.

Original OpenHub MUSE repository: https://github.com/openhublln/MUSE

## Features

- Load a folder of images or select individual image files.
- Draw bounding boxes directly on each image.
- Click an existing box to select it.
- Move a selected box by dragging it.
- Delete the selected box or clear all boxes on the current image.
- Save annotated images as PNG files.

Supported image formats:

- JPG / JPEG
- PNG
- GIF
- WebP
- BMP

## Files

| File | Description |
| --- | --- |
| `index.html` | Main web page for the annotation tool. |
| `styles.css` | User interface styling. |
| `app.js` | Image loading, annotation, navigation, and export logic. |

## Usage

No installation is required. This is a static HTML/CSS/JavaScript tool.

Open the tool directly in a browser:

```bash
open index.html
```

Or start a simple local server from this folder:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

## Annotation Workflow

1. Click **Open Folder** to load all supported images from a folder, or click **Select Images** to choose specific images.
2. Select an image from the left sidebar.
3. Drag on the image to draw a bounding box.
4. Click a box to select it.
5. Drag a selected box to move it.
6. Use **Delete Box** to remove the selected box.
7. Use **Clear Image** to remove all boxes from the current image.
8. Click **Save Images** to export the annotated images.

## Output

The tool exports annotated images as PNG files.

Output filenames follow this format:

```text
original_image_name_boxed.png
```

If the browser supports folder writing, such as Chrome or Edge, click **Choose Output Folder** before saving. Otherwise, the browser will download the exported files.

## Keyboard Shortcuts

| Key | Action |
| --- | --- |
| Left Arrow | Previous image |
| Right Arrow | Next image |
| Delete / Backspace | Delete selected box |
| Escape | Deselect current box |

## Browser Notes

- Chrome and Edge support choosing an output folder through the File System Access API.
- Other browsers may still work for annotation, but saving may download files one by one.
- All image processing happens locally in the browser. Images are not uploaded to a server.
