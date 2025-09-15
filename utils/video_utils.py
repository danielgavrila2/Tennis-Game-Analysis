import cv2


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    return frames


def save_video(output_video_frames, output_path):
    if not output_video_frames:
        raise ValueError("No frames to save.")

    height, width, layers = output_video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    for frame in output_video_frames:
        out.write(frame)

    out.release()
