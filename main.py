from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner

import yaml
import numpy as np

# Load configuration from config file
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

INPUT_VIDEO_PATH = config["input_video_path"]
OUTPUT_VIDEO_PATH = config["output_video_path"]
MODEL_PATH = config["model_path"]
TRACKS_STUB_PATH = config["tracks_stub_path"]


def main():
    video_frames = read_video(INPUT_VIDEO_PATH)

    tracker = Tracker(MODEL_PATH)
    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=True, stub_path=TRACKS_STUB_PATH
    )

    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track["bbox"], player_id
            )
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = (
                team_assigner.team_colors[team]
            )
    player_ball_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_ball_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    save_video(output_video_frames, OUTPUT_VIDEO_PATH)


if __name__ == "__main__":
    main()
