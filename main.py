from utils import read_video, save_video
from trackers import Tracker

import yaml

# Load configuration from config file
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

INPUT_VIDEO_PATH = config['input_video_path']
OUTPUT_VIDEO_PATH = config['output_video_path']
MODEL_PATH = config['model_path']
TRACKS_STUB_PATH = config['tracks_stub_path']


def main():
    video_frames = read_video(INPUT_VIDEO_PATH)

    tracker = Tracker(MODEL_PATH)
    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=True, stub_path=TRACKS_STUB_PATH
    )
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    save_video(output_video_frames, OUTPUT_VIDEO_PATH)


if __name__ == "__main__":
    main()
