from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
import datetime
import os
import numpy as np
import mediapipe as mp
import pickle
import copy
import itertools

app = Flask(__name__, template_folder='./templates')

capture = 0
switch = 1
camera = cv2.VideoCapture(0)

try:
    os.mkdir('./shots')
except OSError as error:
    pass

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load the pickled model
with open('ann_model.pkl', 'rb') as f:
    model = pickle.load(f)

def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame

def process_photo(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Create a Hands object
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5,max_num_hands=2) as hands:
        # Process the frame using the Hands object
        results = hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                normalized_keypoints = pre_process_landmark(image, hand_landmarks)

                # Prepare the sequence
                sequence = np.array(normalized_keypoints).flatten().reshape(1, -1)

                # Make predictions using your model
                prediction = model.predict(sequence)

                # Define your actions for interpretation
                actions = np.array(['Hello', 'Bathroom', 'My', 'Please', 'Fine', 'Name', 'Yes', 'What', 'You', 'Where','nice','meet','go','busy','have','work','again and repeat','understand','no','take care'])

                # Determine the predicted action
                predicted_action = actions[np.argmax(np.squeeze(prediction))]
                print(predicted_action)
                return predicted_action

    return "No hand detected"

def pre_process_landmark(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    temp_landmark_list = copy.deepcopy(landmark_point)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]

    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))

    normalized_landmarks = [point / max_value for point in temp_landmark_list]
    return normalized_landmarks

def gen_frames():
    global capture
    while True:
        success, frame = camera.read()
        if success:
            if capture:
                now = datetime.datetime.now()
                frame = cv2.flip(frame,flipCode=1)
                photo = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":", ''))])
                cv2.imwrite(photo, frame)
                prediction = process_photo(frame)
                yield f"Prediction: {prediction}"
                capture = 0  # Reset capture flag after processing the frame

            processed_frame = process_frame(frame)
            try:
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/proceed', methods=['POST'])
def proceed():
    global capture
    capture = 1  # Set capture flag to 1 to indicate that the frame needs to be captured for processing
    return redirect(url_for('sign_language_recognition'))


@app.route('/sign_language_recognition')
def sign_language_recognition():
    return render_template('index_camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1
        elif request.form.get('stop') == 'Stop/Start':
            if switch == 1:
                switch = 0
                camera.release()
                cv2.destroyAllWindows()
            else:
                camera = cv2.VideoCapture(0)
                switch = 1
    elif request.method == 'GET':
        return render_template('talk_code.html')
    return render_template('talk_code.html')

@app.route('/get_prediction', methods=['GET'])
def get_prediction():
    global camera
    _, frame = camera.read()
    prediction = process_photo(frame)
    return jsonify(prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

camera.release()
cv2.destroyAllWindows()
