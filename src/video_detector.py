import cv2 
import numpy as np 
import argparse 
 
def detect_in_video(query_path, video_path): 
    # Read query image 
    query_img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE) 
 
    # Initialize SIFT 
    sift = cv2.SIFT_create() 
 
    # Detect keypoints and descriptors for query image 
    kp1, des1 = sift.detectAndCompute(query_img, None) 
 
    # Open video 
    cap = cv2.VideoCapture(video_path) 
    if not cap.isOpened(): 
        print("Error: Could not open video.") 
        return 
 
    while cap.isOpened(): 
        ret, frame = cap.read() 
        if not ret: 
            break 
 
        # Convert frame to grayscale 
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
 
        # Detect keypoints and descriptors in frame 
        kp2, des2 = sift.detectAndCompute(frame_gray, None) 
 
        # Match descriptors using FLANN 
        flann = cv2.FlannBasedMatcher({"algorithm": 0, "trees": 5}, {"checks": 50}) 
        matches = flann.knnMatch(des1, des2, k=2) 
 
        # Apply Lowe's ratio test 
        good_matches = [] 
        for m, n in matches: 
            if m.distance < 0.7 * n.distance: 
                good_matches.append(m) 
 
        # Find homography if enough matches 
        if len(good_matches) > 10: 
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) 
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) 
 
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) 
 
            if M is not None: 
                h, w = query_img.shape 
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2) 
                dst = cv2.perspectiveTransform(pts, M) 
                frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3) 
 
        # Display frame 
        cv2.imshow("Video Detection", frame) 
        if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit 
            break 
 
    cap.release() 
    cv2.destroyAllWindows() 
 
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="SIFT Video Object Detection") 
    parser.add_argument("--query", required=True, help="Path to query image") 
    parser.add_argument("--video", required=True, help="Path to video file") 
    args = parser.parse_args() 
 
    detect_in_video(args.query, args.video) 
