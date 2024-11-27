# Vehicle Counter

A Python-based vehicle counting system using YOLOv8 object detection.

## Description

This project implements a vehicle counter that tracks vehicles entering and leaving an area based on their position over time. It uses OpenCV for video processing, YOLOv8 for object detection, and saves counts to CSV and JSON files.

## Features

- Real-time vehicle tracking across video frames
- Entering/leaving counts with timestamps 
- Persistent tracking between frames
- Saves counts to CSV and JSON files
- Configurable counting line position

## Installation

To install the required dependencies:

bash$: pip install -r requirements.txt


Make sure to set up your VIDEO_SOURCE environment variable pointing to your input video file.

## Usage


This will start the vehicle counter and display the output video.

## Technologies Used

- Python 3.7+
- OpenCV (cv2)
- Ultralytics YOLOv8
- NumPy (np)
- JSON
- CSV
- Python-dotenv

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the YOLOv8 example code from Ultralytics
- Uses OpenCV for video processing
- Saves data using standard Python libraries
