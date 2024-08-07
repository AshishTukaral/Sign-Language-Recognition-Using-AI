# Sign Language Recognition Using AI

Welcome to the Sign Language Recognition project! This project aims to bridge the communication gap between sign language users and the broader community by providing a platform to translate hand signs into text and speech using Artificial Neural Networks (ANN) and various Python libraries.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [System Architecture](#system-architecture)
- [License](#license)

## Introduction

In an increasingly interconnected world, effective communication is fundamental. However, for individuals with hearing impairments, traditional means of communication such as speech may not be accessible. This project introduces a groundbreaking Sign Language Recognition System (SLRS) deployed as a web application, providing a platform where users can seamlessly translate hand signs into both text and speech.

Through the convergence of machine learning, web development, and a commitment to inclusivity, this project strives to empower individuals with hearing impairments by fostering greater understanding and communication within society.

## Features

- Real-time hand sign recognition and translation to text and speech
- High accuracy in recognizing American Sign Language (ASL) gestures
- User-friendly web interface
- Supports different signing styles and individual expressions
- Enhances communication efficiency and accessibility

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sign-language-recognition.git
   cd sign-language-recognition
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the web application:
   ```bash
   python app.py
   ```

## Usage

1. Launch the web application by running the above command.
2. Use a webcam to capture hand signs.
3. The application will translate the hand signs into text and speech in real-time.

## Project Structure

```
.
├── static
│   └── js
├── templates
│   ├── index.html
│   ├── index_camera.html
│   └── talk_code.html
├── labels.csv
├── data_using_multihand.py
├── model.ipynb
├── train_model.pkl
└── app.py
```

- **static/js**: Contains JavaScript files for the web application.
- **templates**: HTML templates for the web application.
- **labels.csv**: Labels for classification.
- **data_using_multihand.py**: Captures images, processes them, and extracts hand coordinates using MediaPipe, saving the data to a CSV file.
- **model.ipynb**: Jupyter notebook for building and training the neural network model.
- **train_model.pkl**: Pickled trained model.
- **app.py**: Main Python file containing Flask code to run the web application and make predictions.

## Data Collection with `data_using_multihand.py`

The `data_using_multihand.py` script is used to capture hand sign images, process them, and save the hand keypoints to a CSV file for training the model. Here’s a brief overview of its functionality:

### Key Functions:

- **process_frame(frame, hands)**: Converts the BGR image to RGB, processes it with MediaPipe, and draws landmarks on the image.
- **calc_bounding_rect(image, landmarks)**: Calculates the bounding rectangle for hand landmarks.
- **pre_process_landmark(image, landmarks)**: Normalizes hand landmarks for model input.

### Usage:

1. **Run the script**:
   ```bash
   python data_using_multihand.py
   ```

2. **Capture images**:
   - The script will open the webcam and start capturing images.
   - Press keys `0-9` to label the images with corresponding numbers.
   - The hand keypoints will be saved to a CSV file specified in the script.

### Handling Multiple Label Files:

For a large number of labels, follow these steps:

1. **Create multiple CSV files**:
   - For 20 labels, create one file for labels `0-9` and another for labels `10-19`.
   - Example:
     - `data1.csv` for labels `0-9`
     - `data2.csv` for labels `10-19`

2. **Merge the files**:
   - Use the `pd.merge` method in Python to combine the data:
     ```python
     import pandas as pd

     data1 = pd.read_csv('data1.csv')
     data2 = pd.read_csv('data2.csv')
     combined_data = pd.concat([data1, data2])
     combined_data.to_csv('combined_data.csv', index=False)
     ```

## System Architecture

### Software Requirements

- Python 3.x
- Python libraries: OpenCV, MediaPipe, TensorFlow, NumPy
- IDE: PyCharm or any other preferred code editor
- Hardware: Laptop or PC with an external webcam for better pixel quality

### Main Libraries

- **OpenCV**: Used for image processing and feature detection.
- **MediaPipe**: Utilized for real-time hand tracking and gesture analysis.
- **TensorFlow**: Employed for building and training neural network models.
- **NumPy**: Used for numerical computations and data manipulation.

### Data Flow

1. **Data Collection**: Capture hand sign images using a webcam.
2. **Preprocessing**: Process images to extract relevant features.
3. **Model Training**: Train the neural network model using TensorFlow.
4. **Real-time Recognition**: Use the trained model to recognize hand signs in real-time and translate them into text and speech.

### Code Overview

The main Flask application is in `app.py`, which includes routes to render templates, capture frames, and process images to make predictions. Here's a brief overview of the main parts of the code:

- **process_frame(frame)**: Processes a video frame to detect hand landmarks and draw them.
- **process_photo(frame)**: Captures an image, processes it to detect hand landmarks, and makes a prediction using the trained model.
- **pre_process_landmark(image, landmarks)**: Normalizes hand landmarks for model input.
- **gen_frames()**: Generates frames for video streaming and captures photos for prediction.
- **Flask Routes**:
  - `/`: Renders the home page.
  - `/proceed`: Sets the capture flag to process the next frame.
  - `/sign_language_recognition`: Renders the camera interface.
  - `/video_feed`: Streams video frames to the web page.
  - `/requests`: Handles capture and stop/start requests.
  - `/get_prediction`: Returns the predicted sign as JSON.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

