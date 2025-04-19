import cv2 
import numpy as np 
import argparse 
 
def detect_and_match(query_path, target_path): 
    # Read images 
    query_img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE) 
    target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE) 
 
    # Initialize ORB 
    orb = cv2.ORB_create() 
 
    # Detect keypoints and descriptors 
    kp1, des1 = orb.detectAndCompute(query_img, None) 
    kp2, des2 = orb.detectAndCompute(target_img, None) 
 
    # Match descriptors using BFMatcher with Hamming distance 
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) 
    matches = bf.match(des1, des2) 
 
    # Sort matches by distance 
    matches = sorted(matches, key=lambda x: x.distance) 
 
    # Draw top matches 
    result = cv2.drawMatches(query_img, kp1, target_img, kp2, matches[:50], None, 
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 
 
    # Save or display result 
    cv2.imwrite("result_orb.jpg", result) 
    cv2.imshow("Matches", result) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
 
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="ORB Feature Detection and Matching") 
    parser.add_argument("--query", required=True, help="Path to query image") 
    parser.add_argument("--target", required=True, help="Path to target image") 
    args = parser.parse_args() 
 
    detect_and_match(args.query, args.target) 
