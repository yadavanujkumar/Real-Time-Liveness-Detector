"""
Passive Liveness Detection System

A real-time liveness detection system that uses blink detection to verify
that a real person is present, defending against deepfakes and spoofing attacks.

Author: Computer Vision Engineer
License: MIT
"""

import time
from collections import deque
from enum import Enum

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist


class LivenessState(Enum):
    """States for the anti-spoofing workflow."""
    SCANNING = 1
    VERIFIED = 2
    SPOOF = 3


# MediaPipe Face Mesh landmark indices for eyes
# Left eye landmarks (from the subject's perspective)
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
# Right eye landmarks (from the subject's perspective)
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# EAR threshold for blink detection
EAR_THRESHOLD = 0.25

# Number of blinks required for verification
BLINKS_REQUIRED = 2

# Timeout for spoof detection (in seconds)
SPOOF_TIMEOUT = 5.0

# Consecutive frames for blink detection
CONSEC_FRAMES_THRESHOLD = 2


def calculate_ear(eye_landmarks: np.ndarray) -> float:
    """
    Calculate the Eye Aspect Ratio (EAR) for a given eye.

    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)

    Where p1-p6 are the 6 landmarks of the eye:
    - p1, p4: horizontal points (left and right corners)
    - p2, p3: upper vertical points
    - p5, p6: lower vertical points

    Args:
        eye_landmarks: Array of shape (6, 2) containing the eye landmark coordinates

    Returns:
        The Eye Aspect Ratio value
    """
    # Compute the euclidean distances between the vertical eye landmarks
    vertical_dist_1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    vertical_dist_2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])

    # Compute the euclidean distance between the horizontal eye landmarks
    horizontal_dist = dist.euclidean(eye_landmarks[0], eye_landmarks[3])

    # Avoid division by zero
    if horizontal_dist == 0:
        return 0.0

    # Calculate and return the EAR
    ear = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist)
    return ear


def get_eye_landmarks(face_landmarks, indices: list,
                      frame_shape: tuple) -> np.ndarray:
    """
    Extract eye landmarks from MediaPipe face landmarks.

    Args:
        face_landmarks: MediaPipe face landmarks object
        indices: List of landmark indices for the eye
        frame_shape: Shape of the video frame (height, width)

    Returns:
        Array of shape (6, 2) containing the eye landmark coordinates
    """
    height, width = frame_shape[:2]
    eye_points = []

    for idx in indices:
        landmark = face_landmarks.landmark[idx]
        x = landmark.x * width
        y = landmark.y * height
        eye_points.append([x, y])

    return np.array(eye_points)


def draw_ear_graph(frame: np.ndarray, ear_history: deque, graph_position: tuple,
                   graph_size: tuple = (200, 100)) -> None:
    """
    Draw a real-time EAR graph in the corner of the video feed.

    Args:
        frame: The video frame to draw on
        ear_history: Deque containing historical EAR values
        graph_position: (x, y) position of the graph's top-left corner
        graph_size: (width, height) of the graph
    """
    graph_width, graph_height = graph_size
    x_offset, y_offset = graph_position

    # Create semi-transparent background for the graph
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_offset, y_offset),
                  (x_offset + graph_width, y_offset + graph_height),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Draw graph border
    cv2.rectangle(frame, (x_offset, y_offset),
                  (x_offset + graph_width, y_offset + graph_height),
                  (255, 255, 255), 1)

    # Draw threshold line
    threshold_y = int(y_offset + graph_height -
                      (EAR_THRESHOLD * graph_height * 2))
    cv2.line(frame, (x_offset, threshold_y),
             (x_offset + graph_width, threshold_y), (0, 0, 255), 1)
    cv2.putText(frame, "TH", (x_offset + graph_width - 20, threshold_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    # Draw EAR graph line
    if len(ear_history) > 1:
        points = []
        for i, ear_val in enumerate(ear_history):
            x = x_offset + int((i / (len(ear_history) - 1))
                               * (graph_width - 1))
            # Normalize EAR value (typical range 0.15 - 0.4)
            normalized_ear = min(max(ear_val, 0), 0.5)
            y = int(y_offset + graph_height -
                    (normalized_ear * graph_height * 2))
            y = max(y_offset, min(y, y_offset + graph_height))
            points.append((x, y))

        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], (0, 255, 0), 2)

    # Draw graph title
    cv2.putText(frame, "EAR Graph", (x_offset + 5, y_offset + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def draw_status_overlay(frame: np.ndarray, state: LivenessState,
                        blink_count: int, ear: float) -> None:
    """
    Draw the status overlay on the video frame.

    Args:
        frame: The video frame to draw on
        state: Current liveness detection state
        blink_count: Number of blinks detected
        ear: Current EAR value
    """
    height, width = frame.shape[:2]

    # Define colors and messages based on state
    if state == LivenessState.SCANNING:
        color = (0, 165, 255)  # Orange
        message = "Please Blink to Verify"
        status = f"Blinks: {blink_count}/{BLINKS_REQUIRED}"
    elif state == LivenessState.VERIFIED:
        color = (0, 255, 0)  # Green
        message = "REAL PERSON DETECTED"
        status = "Verification Complete"
    else:  # SPOOF
        color = (0, 0, 255)  # Red
        message = "SPOOF / DEEPFAKE SUSPECTED"
        status = "No blink detected in 5 seconds"

    # Draw border around the frame
    cv2.rectangle(frame, (10, 10), (width - 10, height - 10), color, 3)

    # Draw status box at the top
    cv2.rectangle(frame, (10, 10), (width - 10, 70), color, -1)

    # Draw main message
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = (width - text_size[0]) // 2
    cv2.putText(frame, message, (text_x, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Draw status
    cv2.putText(frame, status, (text_x, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw EAR value
    cv2.putText(frame, f"EAR: {ear:.3f}", (20, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    """Main function to run the liveness detection system."""
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize variables
    blink_count = 0
    consec_frames = 0
    start_time = time.time()
    state = LivenessState.SCANNING
    ear_history = deque(maxlen=50)  # Store last 50 EAR values for graph

    print("=" * 50)
    print("Passive Liveness Detection System")
    print("=" * 50)
    print("Instructions:")
    print("- Look at the camera and blink naturally")
    print(f"- Blink {BLINKS_REQUIRED} times to verify you are a real person")
    print("- Press 'r' to reset the verification")
    print("- Press 'q' to quit")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Face Mesh
        results = face_mesh.process(rgb_frame)

        current_ear = 0.0

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # Extract eye landmarks
            left_eye = get_eye_landmarks(
                face_landmarks, LEFT_EYE_INDICES, frame.shape)
            right_eye = get_eye_landmarks(
                face_landmarks, RIGHT_EYE_INDICES, frame.shape)

            # Calculate EAR for both eyes
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            current_ear = (left_ear + right_ear) / 2.0

            # Add EAR to history for graph
            ear_history.append(current_ear)

            # Draw eye contours (optional, for visualization)
            cv2.polylines(
                frame, [
                    left_eye.astype(
                        np.int32)], True, (0, 255, 255), 1)
            cv2.polylines(
                frame, [
                    right_eye.astype(
                        np.int32)], True, (0, 255, 255), 1)

            # Blink detection logic (only in SCANNING state)
            if state == LivenessState.SCANNING:
                if current_ear < EAR_THRESHOLD:
                    consec_frames += 1
                else:
                    if consec_frames >= CONSEC_FRAMES_THRESHOLD:
                        blink_count += 1
                        print(f"Blink detected! Total blinks: {blink_count}")

                        if blink_count >= BLINKS_REQUIRED:
                            state = LivenessState.VERIFIED
                            print(">>> REAL PERSON VERIFIED <<<")
                    consec_frames = 0
        else:
            # No face detected
            ear_history.append(0.0)

        # Check for timeout (spoof detection) only in SCANNING state
        if state == LivenessState.SCANNING:
            elapsed_time = time.time() - start_time
            if elapsed_time > SPOOF_TIMEOUT and blink_count < BLINKS_REQUIRED:
                state = LivenessState.SPOOF
                print(">>> SPOOF/DEEPFAKE SUSPECTED <<<")

        # Draw EAR graph in the corner
        frame_height, frame_width = frame.shape[:2]
        graph_position = (frame_width - 210, 80)
        draw_ear_graph(frame, ear_history, graph_position)

        # Draw status overlay
        draw_status_overlay(frame, state, blink_count, current_ear)

        # Display the frame
        cv2.imshow("Liveness Detection", frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset verification
            blink_count = 0
            consec_frames = 0
            start_time = time.time()
            state = LivenessState.SCANNING
            ear_history.clear()
            print(">>> Verification reset <<<")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("Liveness detection stopped.")


if __name__ == "__main__":
    main()
