from utils import get_center, get_distance
from ultralytics import YOLO
import cv2
import pickle
import sys
import numpy as np
sys.path.append("../")


class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            results = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]

            if object_cls_name == "person":
                player_dict[track_id] = results

        return player_dict

    def detect_frames(self, frames, read_from_stubs=False, stub_path=None):
        player_detections = []

        if read_from_stubs and stub_path is not None:
            with open(stub_path, "rb") as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(player_detections, f)

        return player_detections

    def draw_boxes(self, frames, player_detections):
        output_frames = []

        for frame, plater_dict in zip(frames, player_detections):
            # Draw bounding boxes on the frame
            for track_id, box in plater_dict.items():
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"PLAYER_ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            output_frames.append(frame)

        return output_frames

    def filter_players(self, court_keypoints, player_detections):
        pl_detections_1 = player_detections[0]
        chosen_pl = self.choose_players(court_keypoints, pl_detections_1)

        filtered_players = []
        for player_dict in player_detections:
            filtered_dict = {k: v for k,
                             v in player_dict.items() if k in chosen_pl}
            filtered_players.append(filtered_dict)

        return filtered_players

    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center(bbox)

            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = get_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

        # sorrt the distances in ascending order
        distances.sort(key=lambda x: x[1])
        # Choose the first 2 tracks
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players
