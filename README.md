# SIFT Assignment (E-JUST) 
 
This repository implements feature extraction and matching using SIFT or ORB in OpenCV to locate an object from a query image within a target image, as per the E-JUST assignment (April 13, 2025). It includes scripts for SIFT and ORB detectors, with an optional bonus script for video processing. 
 
## Requirements 
- Python 3.8+ 
- OpenCV (`opencv-contrib-python`) 
- NumPy 
 
## Setup 
1. Clone the repository: 
   ```bash 
   git clone https://github.com/your-username/sift-assignment.git 
   cd sift-assignment 
   ``` 
2. Install dependencies: 
   ```bash 
   pip install -r requirements.txt 
   ``` 
3. Place query and target images in the `images/` folder. 
 
## Usage 
- Run SIFT detector: 
  ```bash 
  python src/sift_detector.py --query images/query.jpg --target images/target.jpg 
  ``` 
- Run ORB detector: 
  ```bash 
  python src/orb_detector.py --query images/query.jpg --target images/target.jpg 
  ``` 
- Run video detector (bonus): 
  ```bash 
  python src/video_detector.py --video path/to/video.mp4 
  ``` 
 
## Assignment Details 
- **Objective**: Detect an object from a query image in a target image using SIFT or ORB, drawing keypoints on the detected object. 
- **Bonus**: Track the object in a video and draw a bounding rectangle. 
- **Deadline**: April 20, 2025 
- **Team**: Youssef Elamir, Omar Abdelgawad, Eyad Magdy 
 
## License 
MIT License 
