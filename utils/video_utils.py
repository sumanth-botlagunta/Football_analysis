import cv2


def read_video(video_path):
    """
    Reads a video file and returns its frames.

    Args:
        video_path (str): Path to the video file.

    Returns:
        list: A list of frames from the video.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()

    return frames


def save_video(video_frames, output_path, fps=24, codec="XVID"):
    """
    Saves a list of video frames to a video file.

    Args:
        video_frames (list): List of frames to be saved.
        output_path (str): Path to save the video file.
        fps (int, optional): Frames per second. Defaults to 24.
        codec (str, optional): Codec to use for saving the video. Defaults to 'XVID'.
    """
    if not video_frames:
        raise ValueError("No frames to save")

    frame_height, frame_width = video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        raise IOError(f"Cannot open video writer for file: {output_path}")

    try:
        for frame in video_frames:
            out.write(frame)
    finally:
        out.release()
