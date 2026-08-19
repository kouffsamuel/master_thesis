from annotations.master_source import MasterSource
from ultralytics import YOLO
from annotations.clusters_matching import match_clusters_to_detections, match_radar_clusters_to_lidar


class CameraMaster(MasterSource):
    def __init__(self, yolo_model_path, cam_matrix, dist_coeff, tracker_config="fasttrack.yaml", classes=(0,1,2,3,5,7),
                 max_distance_px=100):
        self.yolo = YOLO(yolo_model_path)
        self.cam_matrix = cam_matrix
        self.dist_coeff = dist_coeff
        self.tracker_config = tracker_config
        self.classes = list(classes)
        self.max_distance_px = max_distance_px

    def detect(self, context: dict) -> list[dict]:
        img = context["img"]
        tracks = self.yolo.track(img, persist=True, tracker=self.tracker_config, classes=self.classes)

        detections = []

        for result in tracks:
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            xywh = boxes.xywh.cpu().numpy()
            names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
            ids = boxes.id.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), track_id, (cx, cy, w, h), name in zip(xyxy, ids, xywh, names):
                detections.append({
                    "id": int(track_id),
                    "center": (float(cx), float(cy)),
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "width": float(w),
                    "height": float(h),
                    "class": name,
                })
        return detections

    def match_lidar_clusters(self, clusters_lidar: list[dict], detections: list[dict]) -> list[dict]:
        if not detections or not clusters_lidar:
            return clusters_lidar

        return match_clusters_to_detections(
            clusters_lidar, detections, self.cam_matrix, self.dist_coeff,
            max_distance_px=self.max_distance_px
        )
