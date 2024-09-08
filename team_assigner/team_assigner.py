from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self) -> None:
        """
        Initializes the TeamAssigner with empty dictionaries for team colors and player-team assignments.
        """
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None

    def get_clustering_model(self, image: np.ndarray) -> KMeans:
        """
        Fits a KMeans clustering model to the given image.

        Args:
            image (np.ndarray): The image to fit the clustering model on.

        Returns:
            KMeans: The fitted KMeans model.
        """
        image_2d = image.reshape(-1, 3)
        k_means = KMeans(n_clusters=2, init='k-means++', n_init=2)
        k_means.fit(image_2d)
        return k_means

    def get_player_color(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """
        Extracts the dominant color of a player from the given frame and bounding box.

        Args:
            frame (np.ndarray): The frame containing the player.
            bbox (list): The bounding box coordinates [x1, y1, x2, y2] of the player.

        Returns:
            np.ndarray: The dominant color of the player.
        """
        if len(bbox) != 4:
            raise ValueError("Bounding box must contain exactly four elements [x1, y1, x2, y2].")

        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[0:int(image.shape[0] / 2), :]

        k_means = self.get_clustering_model(top_half_image)
        labels = k_means.labels_

        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster
        player_color = k_means.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame: np.ndarray, player_detections: dict) -> None:
        """
        Assigns team colors based on the detected players in the frame.

        Args:
            frame (np.ndarray): The frame containing the players.
            player_detections (dict): A dictionary of player detections with bounding boxes.
        """
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame: np.ndarray, player_bbox: list, player_id: int) -> int:
        """
        Determines the team of a player based on their color.

        Args:
            frame (np.ndarray): The frame containing the player.
            player_bbox (list): The bounding box coordinates [x1, y1, x2, y2] of the player.
            player_id (int): The ID of the player.

        Returns:
            int: The team ID of the player.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1

        self.player_team_dict[player_id] = team_id
        return team_id
    

