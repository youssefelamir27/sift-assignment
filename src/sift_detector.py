import cv2 
import numpy as np 
import argparse 
 
def detect_and_match(query_path, target_path): 
    # Read images 
    query_img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE) 
    target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE) 
 
    # Initialize SIFT 
    sift = cv2.SIFT_create() 
 
    # Detect keypoints and descriptors 
    kp1, des1 = sift.detectAndCompute(query_img, None) 
    kp2, des2 = sift.detectAndCompute(target_img, None) 
 
    # Match descriptors using FLANN matcher 
    flann = cv2.FlannBasedMatcher({"algorithm": 0, "trees": 5}, {"checks": 50}) 
    matches = flann.knnMatch(des1, des2, k=2) 
 
    # Apply Lowe's ratio test 
    good_matches = [] 
    for m, n in matches: 
        if m.distance < 0.7 * n.distance: 
            good_matches.append(m) 
 
    # Draw keypoints on target image 
    result = cv2.drawMatches(query_img, kp1, target_img, kp2, good_matches, None, 
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 
 
    # Save or display result 
    cv2.imwrite("result.jpg", result) 
    cv2.imshow("Matches", result) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
 
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="SIFT Feature Detection and Matching") 
    parser.add_argument("--query", required=True, help="Path to query image") 
    parser.add_argument("--target", required=True, help="Path to target image") 
    args = parser.parse_args() 
 
    detect_and_match(args.query, args.target) 
