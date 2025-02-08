# Face Filter Application

## Overview

Face Filter Application is a real-time face filter application that leverages OpenCV, MediaPipe, and Hydra Lightning Framework to apply augmented reality (AR) effects to human faces. The application enables users to overlay virtual filters, masks, and effects on their faces in real time using a webcam.

## Features

**Real-time Face Detection**: Uses MediaPipe Face Mesh for precise facial landmark detection.

**Filter Overlay**: Apply custom filters, masks, and augmented reality (AR) effects.

**Modular Configuration**: Uses Hydra for flexible configuration management.

**Efficient Processing**: Optimized with Lightning for improved performance.

**Easy Integration**: Open-source and customizable for various applications.

##Tech Stack

**OpenCV**: For image processing and real-time video streaming.

**MediaPipe**: For face detection and landmark extraction.

**Hydra**: For managing configuration settings dynamically.

**Lightning**: For scalable and efficient deep learning model deployment (if applicable).

## Installation

### Prerequisites

Ensure you have Python 3.8+ installed on your system.

### Steps

Clone the repository:
```bash
git clone https://github.com/phulocnguyen/Face-Filter-Application.git
cd Face-Filter-Application
```
Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
Install dependencies:
```bash
pip install -r requirements.txt
```
## Usage

Run the application with:
```bash
python src/main.py
```
## Configurations

The application uses Hydra for dynamic configuration management. You can modify settings via YAML files in the configs/ directory.
```bash
python main.py --config-name default

Example: Changing the filter type via command line

python main.py filter.type=glasses
```
