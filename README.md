# Automatic License Plate Recognition using YOLOv8

## Overview
This project detects vehicles and their license plates in videos using YOLOv8, recognizes license numbers with OCR, and tracks vehicles across frames for robust results.

## Algorithm Flow

```
Video Input
   |
   v
[Frame Extraction]
   |
   v
[Vehicle Detection (YOLOv8)]
   |
   v
[Vehicle Tracking (SORT)]
   |
   v
[License Plate Detection (YOLOv8, fine-tuned)]
   |
   v
[License Plate Cropping]
   |
   v
[OCR Recognition (EasyOCR)]
   |
   v
[Result Storage (CSV)]
   |
   v
[Interpolation & Smoothing]
   |
   v
[Visualization]
```

## Model Details

- **Vehicle Detection:** Pre-trained YOLOv8n model
- **License Plate Detection:** YOLOv8 fine-tuned on [this dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4) ([model weights](https://drive.google.com/file/d/1p8m7nsRvJJGQvmavFLhPUvGF-0GPfS3H/view?usp=sharing))
- **Tracking:** SORT algorithm
- **OCR:** EasyOCR

## Installation

1. **Clone the repository**
   ```bash
   git clone <repo_url>
   cd Automatic-License-Plate-Recognition-using-YOLOv8
   ```

2. **Create and activate a Python 3.10 environment**
   ```bash
   conda create --prefix ./env python=3.10 -y
   conda activate ./env
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run detection and tracking on the sample video**
   ```bash
   python main.py
   ```
   - Outputs: `test.csv` with detection and recognition results.

2. **Interpolate missing data for smooth tracking**
   ```bash
   python add_missing_data.py
   ```
   - Outputs: `test_interpolated.csv`

3. **Visualize results**
   ```bash
   python visualize.py
   ```
   - Uses the interpolated CSV for smooth output.

## File Structure

- `main.py`: Runs detection, tracking, and recognition
- `add_missing_data.py`: Interpolates missing data for smooth results
- `visualize.py`: Visualizes detection and tracking results
- `requirements.txt`: Python dependencies
- `license_plate_detector.pt`, `yolov8n.pt`: Model weights
- `vid.mp4`: Sample input video

## Notes

- Make sure you have the model weights in the project directory.
- For custom videos, replace `vid.mp4` with your own file.
- Results are saved as CSV files for further analysis or visualization.
