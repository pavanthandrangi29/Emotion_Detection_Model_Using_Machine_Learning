# Emotion_Detection_Model_Using_Machine_Learning

![collage beside](https://github.com/user-attachments/assets/eee1a2ed-f4d1-4ffe-9bf4-e63d3a126643)
## Working Video
https://github.com/user-attachments/assets/8af9e29c-e7bc-491a-a9db-f4539a8580e7
## Interface for detecting emotions from local files
![Screenshot (168)](https://github.com/user-attachments/assets/a85ef76a-4d80-4109-ad46-ab1c5d5867a5)

## Uploading Image
![Screenshot (169)](https://github.com/user-attachments/assets/5c88c49f-6b23-49e8-980e-a09d808c77b6)
## Detecting Enotion
![Screenshot (170)](https://github.com/user-attachments/assets/1f59363e-c5b4-4272-921d-5cbe42fadcb1)
## Uploading Image
![Screenshot (171)](https://github.com/user-attachments/assets/1f00da7d-7fb5-4d26-bc5a-c3311bd6f365)
## Detecting Enotion
![Screenshot (172)](https://github.com/user-attachments/assets/d7d9596d-925a-4391-a3a1-652b73670b79)


An Emotion Detection application using facial recognition and deep learning techniques. This project utilizes OpenCV, TensorFlow, and Keras to detect human emotions from images or real-time webcam input.

## Features

- **Emotion Detection**: Detects emotions like Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
- **Image Upload**: Upload images to detect emotions.
- **Emotion Detection from Images**: Upload images to detect emotions from local files.
- **Real-Time Detection**: Detect emotions from webcam feed in real-time.

## Installation
pip install -r requirements.txt


### Prerequisites

- Python 3.10 
  

### Clone the Repository

``sh
git clone https://github.com/your-username/emotion-detector.git
cd emotion-detector
## GUI Controls

### Emotion Detection from Local Files
- **Upload Image**: Click the "Upload Image" button to select an image file from your device.
- **Detect Emotion**: After uploading an image, click the "Detect Emotion" button to see the detected emotion.

### Real-Time Detection Controls
- **Quit**: Press 'q' to quit the real-time emotion detection.

## How It Works

1. **Face Detection**: The application uses OpenCV's Haar Cascade Classifier to detect faces in images or video frames.
2. **Feature Extraction**: The detected face region is resized to 48x48 pixels and normalized.
3. **Emotion Prediction**: The pre-trained model predicts the emotion label from the extracted features.
