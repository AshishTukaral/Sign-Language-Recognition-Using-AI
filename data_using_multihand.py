import cv2
import mediapipe as mp
import csv
import numpy as np
import copy
import itertools


# Initialize MediaPipe MultiHand
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def process_frame(frame, hands):
    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame
    results = hands.process(frame_rgb)
    # Draw landmarks on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def pre_process_landmark(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    # Normalize landmarks
    temp_landmark_list = copy.deepcopy(landmark_point)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]

    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list and normalize
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))

    normalized_landmarks = [point / max_value for point in temp_landmark_list]

    return normalized_landmarks

def main(csv_filename):
    cap = cv2.VideoCapture(0)  # Open default camera

    # Initialize MediaPipe MultiHand
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5,max_num_hands=2) as hands:
        # Open the CSV file in append mode
        with open(csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process the frame
                frame = cv2.flip(frame, flipCode=1)
                processed_frame = process_frame(frame, hands)

                cv2.imshow('Frame', processed_frame)

                key = cv2.waitKey(1) & 0xFF
                if key >= ord('0') and key <= ord('9'):
                    # Extract and preprocess keypoints
                    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            normalized_keypoints = pre_process_landmark(frame, hand_landmarks)

                            # Create keypoints array with the pressed key as the 0th value
                            keypoints_with_label = [key - ord('0')] + normalized_keypoints

                            # Write keypoints with label to the CSV file
                            csv_writer.writerow(keypoints_with_label)

                if key == ord('q'):
                    break


            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    csv_filename = "../pythonProject1432/new/hand_keypoints_normalized40.csv"  # Name of the CSV file to save all normalized hand keypoints10
    main(csv_filename)

