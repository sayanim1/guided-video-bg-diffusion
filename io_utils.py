import cv2
import numpy as np

def read_video(path):
    """
    Read an RGB video from the specified path.
    Returns:
        frames: list of HxWx3 uint8 numpy arrays in RGB format.
        fps: frames per second of the video.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames, fps

def write_video(frames, path, fps):
    """
    Write a list of RGB frames to an MP4 video file.
    frames: list of HxWx3 uint8 numpy arrays in RGB format.
    path: output video file path.
    fps: frames per second.
    """
    if not frames:
        return
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    for frame in frames: # Expects RGB
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    out.release()
