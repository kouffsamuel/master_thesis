from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from processing.radar_parameters import velocity_bins, range_bins, N
from processing.lidar_processing import COLOR_MODE, DISPLAY_HFOV, DISPLAY_VFOV, NEAR, PITCH, POINT_SIZE, ROLL, YAW, IMAGE_HEIGHT, IMAGE_WIDTH,  project, colorize

CMAP = plt.get_cmap("tab20")

class RealTimeViewer:
    """
    panels: windows to display according to : 
    'rd'      -> Range-Doppler map
    'cluster' -> clusters DBSCAN on RD map
    'track'   -> tracking history
    'cam'     -> cam + bboxes
    'lidar'   -> point cloud LiDAR
    """
    def __init__(self, panels=("rd", "cam")):
        self.paused = False
        self.panels = list(panels)
        self.save_dir = Path("/home/skouff/master_thesis/camera_calibration/new_calibration_files_paper")


        self.ax = {}
        self.im = {}
        self.cbar = {}
        self.title = {}
        self.scatter_lidar = None
        self.mirror_artists = []

        self._build_layout()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.ion()
        plt.show()

    def _build_layout(self):
        has_lidar = "lidar" in self.panels
        others = [p for p in self.panels if p != "lidar"]

        if has_lidar and others:
            # lidar en bas, sur toute la largeur ; le reste en haut
            self.fig = plt.figure(figsize=(6 * len(others), 12))
            gs = GridSpec(2, len(others), figure=self.fig,
                          height_ratios=[1.5, 1.2], hspace=0.25, wspace=0.15)
            for i, p in enumerate(others):
                self.ax[p] = self.fig.add_subplot(gs[0, i])
            self.ax["lidar"] = self.fig.add_subplot(gs[1, :])
        elif has_lidar:
            self.fig, ax = plt.subplots(figsize=(8, 8))
            self.ax["lidar"] = ax
        else:
            self.fig, axes = plt.subplots(1, len(others), figsize=(6 * len(others), 6))
            axes = [axes] if len(others) == 1 else axes
            self.ax = dict(zip(others, axes))

        for p, ax in self.ax.items():
            if p in ("rd", "cluster", "track"):
                ax.set_xlabel("Velocity (km/h)")
                ax.set_ylabel("Range (m)")
                ax.set_xlim(velocity_bins[0], velocity_bins[-1])
                ax.set_ylim(range_bins[0], range_bins[-1])
                self.title[p] = ax.set_title("")
                if p == "track":
                    ax.set_title("Range-Doppler Track History")
                    ax.grid(True, alpha=0.3)
            elif p == "cam":
                ax.axis("off")
            elif p == "lidar":
                ax.axis("off")
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True)

    def _save_fig(self):
        self.save_counter += 1
        filename = os.path.join(self.save_dir, f"frame_{self.save_counter:04d}.png")
        self.fig.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {filename}")

    def on_key(self, event):
        if event.key == 'p':
            self.paused = not self.paused
            print("Paused" if self.paused else "Resuming")
        elif event.key == 'f':
            self._save_fig()
        elif event.key == 'q':
            print("Quitting...")
            plt.close('all')
            os._exit(0)

    def update(self, **data):
        if "rd" in self.ax and "rd_power" in data:
            self.render_rd(data["times"], data["rd_power"])

        if "cluster" in self.ax and "clusters" in data:
            self.render_cluster(data["peaks"], data["clusters"], data.get("frame_idx", 0))

        if "track" in self.ax and "simple_tracks" in data:
            self.render_tracking_history(data["simple_tracks"])

        if "cam" in self.ax and "img" in data:
            self.render_camera(data.get("camera_times"), data["img"], data.get("bboxes_data", []))

        if "lidar" in self.ax and "pts" in data:
            self.render_lidar(data["pts"], data.get("lidar_time"))

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        while self.paused:
            plt.pause(0.05)

    def render_rd(self, t, rd_power):
        ax = self.ax["rd"]
        rd_db = 10 * np.log10(rd_power)
        if "rd" not in self.im:
            self.im["rd"] = ax.imshow(
                rd_db.T,
                extent=[velocity_bins[0], velocity_bins[-1], range_bins[0], range_bins[-1]],
                origin="lower", cmap="gray_r", aspect="auto", vmin=0,
            )
            self.cbar["rd"] = self.fig.colorbar(self.im["rd"], ax=ax)
            self.cbar["rd"].set_label("dB")
        else:
            self.im["rd"].set_data(rd_db.T)
        self.title["rd"].set_text(f"Radar t = {t:.6f}")

    def render_cluster(self, peaks, clusters, i):
        ax = self.ax["cluster"]
        if "cluster" not in self.im:
            self.im["cluster"] = ax.imshow(
                peaks.T,
                extent=[velocity_bins[0], velocity_bins[-1], range_bins[0], range_bins[-1]],
                origin="lower", cmap="gray_r", aspect="auto", vmin=0, vmax=1,
            )
        else:
            self.im["cluster"].set_data(peaks.T)

        ax.set_title(f"Clusters frame - DBSCAN {i}")
        for collection in ax.collections:
            collection.remove()

        for cluster in clusters:
            xs, ys = [], []
            for d_bin, r_bin in cluster.get("points", []):
                d_bin = int(np.clip(round(d_bin), 0, N - 1))
                r_bin = int(np.clip(round(r_bin), 0, N - 1))
                xs.append(velocity_bins[d_bin])
                ys.append(range_bins[r_bin])
            ax.scatter(xs, ys, c=["red"], s=5, marker="s", alpha=0.6)

            r_bin, d_bin = cluster["centroid"]
            d_bin = int(np.clip(round(d_bin), 0, N - 1))
            r_bin = int(np.clip(round(r_bin), 0, N - 1))
            ax.scatter(velocity_bins[d_bin], range_bins[r_bin], c=["red"], s=80, marker="x", linewidths=2)

    def render_tracking_history(self, simple_tracks):
        ax = self.ax["track"]
        for collection in list(ax.collections):
            collection.remove()
        for line in list(ax.lines):
            line.remove()
        for text in list(ax.texts):
            text.remove()

        for tr_id, centroid_history in simple_tracks:
            color = CMAP(tr_id % CMAP.N)
            ranges, velocities = [], []
            for r_bin, d_bin in centroid_history:
                r_bin = int(np.clip(round(r_bin), 0, N - 1))
                d_bin = int(np.clip(round(d_bin), 0, N - 1))
                ranges.append(range_bins[r_bin])
                velocities.append(velocity_bins[d_bin])

            if len(ranges) >= 2:
                ax.plot(velocities, ranges, "-", color=color, linewidth=2)
            if ranges:
                ax.scatter(velocities[-1], ranges[-1], color=color, s=80)
                ax.text(velocities[-1], ranges[-1], f"ID {tr_id}",
                        color=color, fontsize=10, fontweight="bold")

    def render_camera(self, t, img, bboxes_data):
        ax = self.ax["cam"]
        if "cam" not in self.im:
            self.im["cam"] = ax.imshow(img)
        else:
            self.im["cam"].set_data(img)

        ax.set_title(f"Camera t = {t:.6f}" if t is not None else "Camera")

        for patch in ax.patches:
            patch.remove()
        for text in ax.texts:
            text.remove()

        for x1, y1, x2, y2, label, conf, track_id in bboxes_data:
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                  linewidth=2, edgecolor="lime", zorder=10)
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{label} ID:{track_id} {conf:.2f}",
                    color="white", bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"), zorder=10)

    def render_lidar(self, pts, t=None):
        ax = self.ax["lidar"]
        u, v, forward, keep = project(pts, IMAGE_WIDTH, IMAGE_HEIGHT, DISPLAY_HFOV, DISPLAY_VFOV,
                                       yaw=YAW, pitch=PITCH, roll=ROLL, near=NEAR)
        rgba = colorize(pts, keep, forward, COLOR_MODE)
        size = np.clip(POINT_SIZE * 26.0 / forward, 0.7, 60.0)

        if self.scatter_lidar is None:
            self.scatter_lidar = ax.scatter(u, v, c=rgba, s=size, marker=".", linewidths=0)
            margin = 20
            ax.set_xlim(u.min() - margin, u.max() + margin)
            ax.set_ylim(v.max() + margin, v.min() - margin)
        else:
            offsets = np.column_stack((u, v))
            self.scatter_lidar.set_offsets(offsets)
            self.scatter_lidar.set_facecolor(rgba)
            self.scatter_lidar.set_sizes(size)

        if t is not None:
            ax.set_title(f"LiDAR t = {t:.6f}")