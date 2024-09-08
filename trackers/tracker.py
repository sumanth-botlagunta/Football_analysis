from ultralytics import YOLO
import supervision as sv
import pickle
import os
from utils import get_bbox_center, get_bbox_width
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Any


class Tracker:
    def __init__(self, model_path) -> None:
        """
        Initializes the Tracker with a YOLO model and ByteTrack tracker.

        Args:
            model_path (str): Path to the YOLO model.
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self, ball_positions: List[Dict[int, Dict[str, Any]]]) -> List[Dict[int, Dict[str, List[float]]]]:
        """
        Interpolates missing ball positions in the given list of ball positions.

        Args:
            ball_positions (List[Dict[int, Dict[str, Any]]]): A list of dictionaries containing ball positions with bounding boxes.

        Returns:
            List[Dict[int, Dict[str, List[float]]]]: A list of dictionaries with interpolated ball positions.
        """
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        ball_positions = [{1: {'bbox': bbox}} for bbox in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        """
        Detects objects in the given frames using the YOLO model.

        Args:
            frames (list): List of frames to detect objects in.

        Returns:
            list: List of detections for each frame.
        """
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detection_batch = self.model.predict(frames[i : i + batch_size], conf=0.1)
            detections.extend(detection_batch)
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Gets object tracks from the given frames.

        Args:
            frames (list): List of frames to track objects in.
            read_from_stub (bool, optional): Whether to read tracks from a stub file. Defaults to False.
            stub_path (str, optional): Path to the stub file. Defaults to None.

        Returns:
            dict: Dictionary containing tracks for players, referees, and the ball.
        """
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)
        tracks = {
            "players": [{} for _ in range(len(frames))],
            "referees": [{} for _ in range(len(frames))],
            "ball": [{} for _ in range(len(frames))],
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Update class IDs for goalkeepers
            for object_id, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_id] = cls_names_inv["player"]

            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision
            )

            # Update tracks for players and referees
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif class_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # Update tracks for the ball
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]

                if class_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """
        Draws an ellipse on the given frame.

        Args:
            frame (np.ndarray): The frame to draw on.
            bbox (list): Bounding box coordinates.
            color (tuple): Color of the ellipse.
            track_id (int, optional): ID of the track. Defaults to None.

        Returns:
            np.ndarray: The frame with the ellipse drawn.
        """
        y2 = int(bbox[3])
        x_center, _ = get_bbox_center(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.4 * width)),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = (y2 - rectangle_height // 2) + 15
            y2_rect = (y2 + rectangle_height // 2) + 15

            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        """
        Draws a triangle on the given frame.

        Args:
            frame (np.ndarray): The frame to draw on.
            bbox (list): Bounding box coordinates.
            color (tuple): Color of the triangle.

        Returns:
            np.ndarray: The frame with the triangle drawn.
        """
        y = int(bbox[1])
        x, _ = get_bbox_center(bbox)

        triangle_points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_annotations(self, frames, tracks):
        """
        Draws annotations on the given frames.

        Args:
            frames (list): List of frames to draw annotations on.
            tracks (dict): Dictionary containing tracks for players, referees, and the ball.

        Returns:
            list: List of frames with annotations drawn.
        """
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))
            output_frames.append(frame)
        return output_frames
