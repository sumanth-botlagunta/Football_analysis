from typing import Dict, Any
from utils import get_bbox_center, measure_distance

class PlayerBallAssigner:
    def __init__(self, max_player_ball_distance: int = 70) -> None:
        """
        Initializes the PlayerBallAssigner with a maximum distance for assigning the ball to a player.

        Args:
            max_player_ball_distance (int): The maximum distance allowed for assigning the ball to a player. Default is 70.
        """
        self.max_player_ball_distance = max_player_ball_distance

    def assign_ball_to_player(self, players: Dict[int, Dict[str, Any]], ball_bbox: list) -> int:
        """
        Assigns the ball to the nearest player within the maximum allowed distance.

        Args:
            players (Dict[int, Dict[str, Any]]): A dictionary of players with their bounding boxes.
            ball_bbox (list): The bounding box coordinates [x1, y1, x2, y2] of the ball.

        Returns:
            int: The ID of the assigned player, or -1 if no player is within the maximum allowed distance.
        """
        if not ball_bbox or len(ball_bbox) != 4:
            raise ValueError("ball_bbox must be a list of four elements [x1, y1, x2, y2].")

        ball_position = get_bbox_center(ball_bbox)
        minimum_distance = float('inf')
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player.get('bbox')
            if not player_bbox or len(player_bbox) != 4:
                continue

            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < minimum_distance and distance < self.max_player_ball_distance:
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player
