import os
import numpy as np
import matplotlib.pyplot as plt
from annotations.lidar_projection import project_real_camera, lidar_to_radar_range
import matplotlib.pyplot as plt
from annotations.frame_data import FrameData
from processing.radar_parameters import velocity_bins, range_bins, N

# CAMERA PARAMETERS
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

POINT_SIZE = 15.0
CLUSTER_CMAP = plt.get_cmap("tab20")
UNMATCHED_COLOR = np.array([0.5, 0.5, 0.5, 1.0])

class RealTimeViewer:
    def __init__(self, camMatrix, distCoeff):
        self.camMatrix = camMatrix
        self.distCoeff = distCoeff

        self.paused = False
        self.fig, (self.ax_rd, self.ax_cam) = plt.subplots(1, 2, figsize=(15, 6))

        self.im_rd = None
        self.cbar = None
        self.im_cam = None
        self.scatter_lidar = None
        self.scatter_lidar_points = None

        self.title_rd = self.ax_rd.set_title("")
        self.ax_rd.set_xlabel("Velocity (km/h)")
        self.ax_rd.set_ylabel("Range (m)")
        self.ax_cam.axis("off")

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.ion()
        plt.show()

    def on_key(self, event):
        if event.key == 'p':
            self.paused = not self.paused
            print("Paused" if self.paused else "Resuming")
        elif event.key == 'q':
            print("Quitting...")
            plt.close('all')
            os._exit(0)

    def update(self, frame: FrameData):
        "affiche la frame déjà calculée par le FrameProcessor"
        self._render_radar(frame)
        self._render_camera(frame)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        while self.paused:
            plt.pause(0.05)

    def _render_radar(self, frame: FrameData):
        rd_power = frame.rd_power
        if self.im_rd is None:
            self.im_rd = self.ax_rd.imshow(
                (10 * np.log10(rd_power)).T,
                extent=[velocity_bins[0], velocity_bins[-1], range_bins[0], range_bins[-1]],
                origin='lower', cmap='gray_r', vmin=0, aspect='auto'
            )
            self.cbar = self.fig.colorbar(self.im_rd, ax=self.ax_rd)
            self.cbar.set_label("dB")
        else:
            self.im_rd.set_data((10 * np.log10(rd_power)).T)

        self.title_rd.set_text(f"Radar t = {frame.t_radar:.6f}")
        self._render_cluster_radar(self.ax_rd, frame.clusters_radar)

    def _get_color_for_detection_id(self,det_id):
        if det_id is None:
            return UNMATCHED_COLOR
        return np.array(CLUSTER_CMAP(det_id % CLUSTER_CMAP.N))

    def _render_cluster_radar(self, ax_rd, clusters):
        for collection in ax_rd.collections:
            collection.remove()

        for cluster in clusters:
            det_id = cluster.get("detection_id")
            color = self._get_color_for_detection_id(det_id)

            xs, ys = [], []
            for d_bin, r_bin in cluster.get("points", []):
                d_bin = int(np.clip(round(d_bin), 0, N - 1))
                r_bin = int(np.clip(round(r_bin), 0, N - 1))
                xs.append(velocity_bins[d_bin])
                ys.append(range_bins[r_bin])
            ax_rd.scatter(xs, ys, c=[color], s=5, marker="s", alpha=0.6)

            r_bin, d_bin = cluster["centroid"]
            d_bin = int(np.clip(round(d_bin), 0, N - 1))
            r_bin = int(np.clip(round(r_bin), 0, N - 1))
            ax_rd.scatter(velocity_bins[d_bin], range_bins[r_bin], c=[color], s=80, marker="x", linewidths=2)
        

    def _render_camera(self, frame: FrameData):
        if self.im_cam is None:
            self.im_cam = self.ax_cam.imshow(frame.img)
        else:
            self.im_cam.set_data(frame.img)

        for patch in self.ax_cam.patches:
            patch.remove()
        for text in self.ax_cam.texts:
            text.remove()

        for det in frame.master_detections:
            if "bbox" not in det:
                continue
            x1, y1, x2, y2 = det["bbox"]
            det_id = det["id"]
            self.ax_cam.text(x1, y1 - 5, f"ID:{det_id}",
                              color="white", bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"), zorder=10)
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=1,
                                  edgecolor=self._get_color_for_detection_id(det_id), zorder=10)
            self.ax_cam.add_patch(rect)

        self.ax_cam.set_title(f"Camera t = {frame.t_camera:.6f}")
        self._render_lidar_overlay(frame.clusters_lidar)

    def _render_lidar_overlay(self, clusters_lidar):
        all_u, all_v, all_colors, all_sizes = [], [], [], []
        legend_ids_seen = set()

        for cluster in clusters_lidar:
            u, v, forward, keep = project_real_camera(
                cluster['pcd'], self.camMatrix, self.distCoeff, IMAGE_WIDTH, IMAGE_HEIGHT
            )
            if len(u) == 0:
                continue

            det_id = cluster.get("detection_id")
            color = self._get_color_for_detection_id(det_id)
            rgba = np.tile(color, (len(u), 1))

            fade = np.clip(
                1.05 - 0.35 * (forward / max(np.percentile(forward, 97), 1e-6)),
                0.35, 1.0
            )
            rgba[:, :3] *= fade[:, None]
            size = np.clip(POINT_SIZE * 26.0 / forward, 0.7, 60.0)

            all_u.append(u)
            all_v.append(v)
            all_colors.append(rgba)
            all_sizes.append(size)
            if det_id is not None:
                legend_ids_seen.add(det_id)

        if self.ax_cam.get_legend() is not None:
            self.ax_cam.get_legend().remove()

        if legend_ids_seen:
            handles = [
                plt.Line2D([0], [0], marker='.', color='w',
                           markerfacecolor=self._get_color_for_detection_id(det_id),
                           markersize=10, label=f"ID {det_id}")
                for det_id in sorted(legend_ids_seen)
            ]
            self.ax_cam.legend(
                handles=handles, loc="upper right", fontsize=8, framealpha=0.6,
                labelcolor="white", facecolor="black", edgecolor="none"
            )

        if all_u:
            u_cat = np.concatenate(all_u)
            v_cat = np.concatenate(all_v)
            colors_cat = np.concatenate(all_colors)
            sizes_cat = np.concatenate(all_sizes)

            if self.scatter_lidar is None:
                self.scatter_lidar = self.ax_cam.scatter(
                    u_cat, v_cat, c=colors_cat, s=sizes_cat, marker=".", linewidths=0
                )
            else:
                self.scatter_lidar.set_offsets(np.column_stack((u_cat, v_cat)))
                self.scatter_lidar.set_facecolor(colors_cat)
                self.scatter_lidar.set_sizes(sizes_cat)
        elif self.scatter_lidar is not None:
            self.scatter_lidar.set_offsets(np.empty((0, 2)))

    def _render_lidar_points_overlay(self, pcd_full, color=(0.0, 1.0, 0.4), point_size=8.0):
        """
        Affiche tous les points LiDAR (avant clustering) projetés dans l'image caméra,
        sans distinction d'ID puisqu'aucun matching n'existe à ce niveau.
        """
        if pcd_full is None or pcd_full.is_empty():
            if self.scatter_lidar_points is not None:
                self.scatter_lidar_points.set_offsets(np.empty((0, 2)))
            return

        u, v, forward, keep = project_real_camera(
            pcd_full, self.camMatrix, self.distCoeff, IMAGE_WIDTH, IMAGE_HEIGHT
        )

        if len(u) == 0:
            if self.scatter_lidar_points is not None:
                self.scatter_lidar_points.set_offsets(np.empty((0, 2)))
            return

        base_color = np.array(color + (1.0,))
        rgba = np.tile(base_color, (len(u), 1))

        # Fondu selon la distance, comme pour les clusters
        fade = np.clip(
            1.05 - 0.35 * (forward / max(np.percentile(forward, 97), 1e-6)),
            0.35, 1.0
        )
        rgba[:, :3] *= fade[:, None]
        size = np.clip(point_size * 26.0 / forward, 0.5, 40.0)

        if self.scatter_lidar_points is None:
            self.scatter_lidar_points = self.ax_cam.scatter(
                u, v, c=rgba, s=size, marker=".", linewidths=0, zorder=1  # zorder bas : sous les clusters
            )
        else:
            self.scatter_lidar_points.set_offsets(np.column_stack((u, v)))
            self.scatter_lidar_points.set_facecolor(rgba)
            self.scatter_lidar_points.set_sizes(size)