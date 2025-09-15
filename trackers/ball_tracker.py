from ultralytics import YOLO
import cv2
import pickle
import pandas as pd


class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]

        ball_dict = {}
        for box in results.boxes:
            results = box.xyxy.tolist()[0]
            ball_dict[1] = results

        return ball_dict

    def detect_frames(self, frames, read_from_stubs=False, stub_path=None):
        ball_detections = []

        if read_from_stubs and stub_path is not None:
            with open(stub_path, "rb") as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def draw_boxes(self, frames, player_detections):
        output_frames = []

        for frame, ball_dict in zip(frames, player_detections):
            # Draw bounding boxes on the frame
            for track_id, box in ball_dict.items():
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"BALL_ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            output_frames.append(frame)

        return output_frames

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]

        df_ball_positions = pd.DataFrame(ball_positions, columns=[
                                         'x1', 'y1', 'x2', 'y2']).interpolate().bfill()

        ball_positions = [{1: x}
                          for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def get_ball_shots_frames(self, ball_positions):

        ball_positions = [x.get(1, []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=[
                                         'x1', 'y1', 'x2', 'y2']).interpolate().bfill()

        df_ball_positions['mid_y'] = (
            df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(
            window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff(
        ).fillna(0)
        df_ball_positions['ball_hit'] = 0

        minimum_change_frames_for_hit = 25
        window = int(minimum_change_frames_for_hit * 1.2)

        for i in range(1, len(df_ball_positions) - window):
            neg_changes = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i+1] < 0
            pos_changes = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i+1] > 0

            if neg_changes or pos_changes:
                changes = 0
                for change_frame in range(i+1, i + window + 1):
                    neg_changes_next = df_ball_positions['delta_y'].iloc[
                        i] > 0 and df_ball_positions['delta_y'].iloc[change_frame] < 0
                    pos_changes_next = df_ball_positions['delta_y'].iloc[
                        i] < 0 and df_ball_positions['delta_y'].iloc[change_frame] > 0

                    if neg_changes and neg_changes_next:
                        changes += 1
                    elif pos_changes and pos_changes_next:
                        changes += 1

                if changes >= minimum_change_frames_for_hit:
                    df_ball_positions['ball_hit'].iloc[i] = 1

            frames_ball_was_hit = df_ball_positions.index[df_ball_positions['ball_hit'] == 1].tolist(
            )

        return frames_ball_was_hit
