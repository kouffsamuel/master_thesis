from abc import ABC, abstractmethod

class MasterSource(ABC):
    @abstractmethod
    def detect(self, context:dict) -> list[dict]:
        raise NotImplementedError
    
    @abstractmethod
    def match_lidar_clusters(self, clusters_lidar: list[dict], detections:list[dict]) -> list[dict]:
        raise NotImplementedError