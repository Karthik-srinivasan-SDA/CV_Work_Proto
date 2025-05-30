# CV_Work_Proto
This repository contains the necessary work related to computer vision tasks related to FORGE Proto-sem Project
Requirements:

# 1. Python Dependencies
Package	Purpose
opencv-python	For image processing, camera calibration, display, etc.
numpy	For numerical operations and array manipulation

# 2. System Dependencies (for Raspberry Pi)
Package	Purpose
python3- picamera2	To interface with the Raspberry Pi Camera using Picamera2
libcamera -apps	Required by Picamera2 for camera control

# Pre-Requirements:
- Ensure the Raspberry Pi is properly connected to the camera hardware.
- Enable the camera interface via Raspberry Pi configuration.
- Raspberry Pi OS should be up-to-date.

# Steps to use this Project:

## 1. Clone this repository:

bash
git clone https://github.com/Karthik-srinivasan-SDA/CV_Work_Proto.git

## 2. Locate to the working directory
cd CV_Work_Proto

# 3. Installation Commands

Install Python packages:
bash

pip install opencv-python numpy
Install Raspberry Pi system packages:

bash

sudo apt update
sudo apt install python3-picamera2 libcamera-apps

 Test the scripts availible 






