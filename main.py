from utils import (read_video, save_video)
from trackers import (PlayerTracker, BallTracker)
from court_line_detector import CourtLineDetector
import cv2
from graphical_court import GraphicalCourt


def main():
    # Read video frames
    input_path = "input_videos\input_video.mp4"
    video_frames = read_video(input_path)

    # Initialize player tracker and detect players in frames
    player_tracker = PlayerTracker(model_path="yolov8x")
    player_detections = player_tracker.detect_frames(
        video_frames, read_from_stubs=True, stub_path="tracker_stubs/player_detections.pkl")

    # Initialize ball tracker and detect the ball in frames
    ball_tracker = BallTracker(model_path="models/yolo5_last.pt")
    ball_detections = ball_tracker.detect_frames(
        video_frames, read_from_stubs=True, stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Detect the frames when the ball was hit
    ball_hit_frames = ball_tracker.get_ball_shots_frames(ball_detections)
    print(f"Ball hit frames: {ball_hit_frames}")

    # Detect court lines
    court_line_detector = CourtLineDetector(
        model_path="models/keypoints_model.pth")
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Draw only the players and not the rest of the people
    player_detections = player_tracker.filter_players(
        court_keypoints, player_detections)

    # Draw the bounding boxes on the frames
    output_video_frames = player_tracker.draw_boxes(
        video_frames, player_detections)
    output_video_frames = ball_tracker.draw_boxes(
        output_video_frames, ball_detections)
    output_video_frames = court_line_detector.draw_keypoints_on_frames(
        output_video_frames, court_keypoints)

    # Draw the graphical court
    graphical_court = GraphicalCourt(output_video_frames[0])

    output_video_frames = graphical_court.draw_court_bg(output_video_frames)

    # We will display the frame number on each frame
    for i, frame in enumerate(video_frames):
        cv2.putText(frame, f"Frame: {i+1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # Save the processed video frames
    save_video(output_video_frames, "output_videos\output_video.avi")


if __name__ == "__main__":
    main()
