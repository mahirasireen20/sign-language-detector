# Real Time Sign Language Detection and Voice Output System

## About the Project

This project is a real-time computer-vision based prototype that recognizes a selected set of hand gestures using a standard webcam.

The system uses MediaPipe Hand Landmarker to detect hand landmarks and predefined geometric rules to recognize gestures.

## Technologies Used

- Python
- OpenCV
- MediaPipe
- Math
- Text-to-Speech

## Supported Gestures

### One-Hand Gestures
- MINI HEART
- PEACE
- OK
- TELEPHONE
- I LOVE YOU
- YES
- NO
- WATER

### Two-Hand Gestures
- THANKS
- HEART
- HUG
- HELP

## Project Files

- `sign_language_detector.py` – Main Python source code
- `Sign_Language_report.pdf` – Project report
- `requirements.txt` – Required Python libraries
- `models/hand_landmarker.task` – Hand landmark model
- `README.md` – Project documentation

## Hardware Requirements

- Computer / Laptop
- Webcam
- USB interface

## How It Works

1. Webcam captures the hand gesture.
2. MediaPipe Hand Landmarker detects hand landmarks.
3. Geometric rules analyze the finger positions.
4. The corresponding gesture is identified.
5. The detected word is displayed on the screen.
6. Text-to-Speech can be integrated for voice output.

## Limitations

- Limited gesture vocabulary
- Recognition can be affected by lighting and camera position
- Similar gestures may cause incorrect classification
- Dynamic signs are not fully supported

## Future Enhancements

- More sign vocabulary
- Dynamic sign recognition
- Improved robustness
- Multilingual voice output
- Mobile deployment
- Sentence-level translation
