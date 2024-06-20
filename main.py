from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import time

app = Flask(__name__)

class ArmTracker:
    def __init__(self):
        self.prev_height_left = None
        self.is_moving_left = False
        self.prev_height_right = None
        self.is_moving_right = False
        self.repetitions_entrelazadas = 0
        self.hands_above_head_last = False
        self.repetitions_flexion_left = 0
        self.repetitions_flexion_right = 0
        self.repetitions_lateral_raise_left = 0  # Contador para levantamiento lateral izquierdo
        self.repetitions_lateral_raise_right = 0  # Contador para levantamiento lateral derecho
        self.completed_exercise = False

    def track_arm(self, wrist_y, elbow_y, is_left):
        if is_left:
            if not self.is_moving_left and wrist_y < elbow_y:  # Flexión detectada en brazo izquierdo
                self.is_moving_left = True
                self.prev_height_left = wrist_y
                return True
            elif self.is_moving_left and wrist_y > elbow_y:  # Fin del movimiento en brazo izquierdo
                self.is_moving_left = False
                self.prev_height_left = None
        else:
            if not self.is_moving_right and wrist_y < elbow_y:  # Flexión detectada en brazo derecho
                self.is_moving_right = True
                self.prev_height_right = wrist_y
                return True
            elif self.is_moving_right and wrist_y > elbow_y:  # Fin del movimiento en brazo derecho
                self.is_moving_right = False
                self.prev_height_right = None
        return False

    def track_hands_above_head(self, landmarks):
        if not landmarks:
            return False

        left_wrist_y = landmarks[mp.solutions.holistic.PoseLandmark.LEFT_WRIST].y
        right_wrist_y = landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_WRIST].y
        left_wrist_y = 1 - left_wrist_y
        right_wrist_y = 1 - right_wrist_y

        hands_above_head = left_wrist_y < 0.4 and right_wrist_y < 0.4
        hands_below_head = left_wrist_y > 0.6 and right_wrist_y > 0.6

        if hands_above_head and not self.hands_above_head_last:
            self.repetitions_entrelazadas += 1

        self.hands_above_head_last = hands_above_head

        return hands_above_head or hands_below_head

    def track_lateral_raise(self, landmarks, is_left):
        if not landmarks:
            return False

        shoulder_y = landmarks[mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER].y if is_left else landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER].y
        shoulder_y = 1 - shoulder_y
        raise_threshold = 0.6

        if shoulder_y < raise_threshold and ((is_left and self.prev_height_left is not None and self.prev_height_left > raise_threshold) or (not is_left and self.prev_height_right is not None and self.prev_height_right > raise_threshold)):
            if is_left:
                self.repetitions_lateral_raise_left += 1
            else:
                self.repetitions_lateral_raise_right += 1

        if is_left:
            self.prev_height_left = shoulder_y
        else:
            self.prev_height_right = shoulder_y

        return shoulder_y < raise_threshold

    def check_completed(self):
        return (self.repetitions_flexion_left >= 10 or self.repetitions_flexion_right >= 10 or
                self.repetitions_entrelazadas >= 10 or self.repetitions_lateral_raise_left >= 10 or
                self.repetitions_lateral_raise_right >= 10)

def gen_frames_calentamiento():
    mp_holistic = mp.solutions.holistic
    video_capture = cv2.VideoCapture(0)
    arm_tracker = ArmTracker()
    start_time = time.time()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)

            elapsed_time = time.time() - start_time
            countdown = 10 - int(elapsed_time)

            if countdown > 0:
                cv2.putText(frame, f"Empieza en: {countdown}", (10, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            else:
                if results.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                    left_wrist_y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y * frame.shape[0]
                    left_elbow_y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].y * frame.shape[0]
                    if arm_tracker.track_arm(left_wrist_y, left_elbow_y, True):
                        arm_tracker.repetitions_flexion_left += 1

                    right_wrist_y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y * frame.shape[0]
                    right_elbow_y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].y * frame.shape[0]
                    if arm_tracker.track_arm(right_wrist_y, right_elbow_y, False):
                        arm_tracker.repetitions_flexion_right += 1

                    cv2.putText(frame, f"Flexiones izquierda: {arm_tracker.repetitions_flexion_left}", (10, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, f"Flexiones derecha: {arm_tracker.repetitions_flexion_right}", (10, frame.shape[0] - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if arm_tracker.check_completed():
                arm_tracker.completed_exercise = True
                cv2.putText(frame, "Finalizado", (10, frame.shape[0] - 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()

def gen_frames_entrelazadas():
    mp_holistic = mp.solutions.holistic
    video_capture = cv2.VideoCapture(0)
    arm_tracker = ArmTracker()
    start_time = time.time()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)

            elapsed_time = time.time() - start_time
            countdown = 10 - int(elapsed_time)

            if countdown > 0:
                cv2.putText(frame, f"Empieza en: {countdown}", (10, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            else:
                if results.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                    if arm_tracker.track_hands_above_head(results.pose_landmarks.landmark):
                        cv2.putText(frame, f"Repeticiones entrelazadas: {arm_tracker.repetitions_entrelazadas}", (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if arm_tracker.check_completed():
                arm_tracker.completed_exercise = True
                cv2.putText(frame, "Finalizado", (10, frame.shape[0] - 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()

def gen_frames_lateral_espalda():
    mp_holistic = mp.solutions.holistic
    video_capture = cv2.VideoCapture(0)
    arm_tracker = ArmTracker()
    start_time = time.time()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)

            elapsed_time = time.time() - start_time
            countdown = 10 - int(elapsed_time)

            if countdown > 0:
                cv2.putText(frame, f"Empieza en: {countdown}", (10, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            else:
                if results.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                    if arm_tracker.track_lateral_raise(results.pose_landmarks.landmark, True):
                        cv2.putText(frame, f"Repeticiones levantamiento lateral izquierdo: {arm_tracker.repetitions_lateral_raise_left}", (10, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if arm_tracker.track_lateral_raise(results.pose_landmarks.landmark, False):
                        cv2.putText(frame, f"Repeticiones levantamiento lateral derecho: {arm_tracker.repetitions_lateral_raise_right}", (10, frame.shape[0] - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if arm_tracker.check_completed():
                arm_tracker.completed_exercise = True
                cv2.putText(frame, "Finalizado", (10, frame.shape[0] - 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()

@app.route('/calentamiento')
def calentamiento_feed():
    return Response(gen_frames_calentamiento(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/entrelazadas')
def entrelazadas_feed():
    return Response(gen_frames_entrelazadas(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/lateral_espalda')
def lateral_espalda_feed():
    return Response(gen_frames_lateral_espalda(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
