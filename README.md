# Real-Time Liveness Detector

A **Passive Liveness Detection** system that uses blink detection to verify that a real person is present, defending against deepfakes and spoofing attacks (like holding a photo up to a camera).

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)

## Features

- **Real-time Face Detection**: Uses MediaPipe Face Mesh for accurate facial landmark detection
- **Eye Aspect Ratio (EAR) Calculation**: Monitors eye openness to detect natural blinks
- **Anti-Spoofing Workflow**: Three-state verification system
  - 🟠 **Scanning**: Waiting for user to blink
  - 🟢 **Verified**: Real person detected after natural blinks
  - 🔴 **Spoof/Timeout**: No blink detected within 5 seconds
- **Real-time EAR Graph**: Visual display of EAR values in the corner of the video feed
- **Mirror Display**: Natural viewing experience with horizontally flipped video

## How It Works

The system uses the **Eye Aspect Ratio (EAR)** to detect blinks:

```
EAR = (Vertical Distance) / (Horizontal Distance)
```

When a person blinks, the EAR drops significantly below the threshold (0.25). The system counts natural blinks and verifies the user as a real person after 2-3 blinks. If no blinks are detected within 5 seconds, it flags a potential spoofing attempt.

## Installation

### Prerequisites

- Python 3.9 or higher
- Webcam

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yadavanujkumar/Real-Time-Liveness-Detector.git
cd Real-Time-Liveness-Detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the liveness detector:

```bash
python liveness_detector.py
```

### Controls

- **`r`** - Reset verification
- **`q`** - Quit the application

### Workflow

1. Position your face in front of the webcam
2. The system displays "Please Blink to Verify" with an orange border
3. Blink naturally 2-3 times
4. Upon successful verification, the border turns green with "REAL PERSON DETECTED"
5. If no blinks are detected within 5 seconds, the border turns red with "SPOOF / DEEPFAKE SUSPECTED"

## Technical Details

### Dependencies

- `opencv-python`: Video capture and image processing
- `mediapipe`: Face mesh landmark detection
- `numpy`: Numerical computations
- `scipy`: Distance calculations for EAR

### Key Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `EAR_THRESHOLD` | 0.25 | Threshold below which a blink is detected |
| `BLINKS_REQUIRED` | 2 | Number of blinks needed for verification |
| `SPOOF_TIMEOUT` | 5.0 seconds | Time before flagging as potential spoof |
| `CONSEC_FRAMES_THRESHOLD` | 2 | Consecutive frames to confirm a blink |

### Eye Landmarks

The system uses MediaPipe Face Mesh landmarks:
- **Left Eye**: Indices [362, 385, 387, 263, 373, 380]
- **Right Eye**: Indices [33, 160, 158, 133, 153, 144]

## Use Cases

- Identity verification systems
- Online exam proctoring
- Cybersecurity demonstrations
- Anti-spoofing for access control

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Computer Vision Engineer specializing in Identity Verification