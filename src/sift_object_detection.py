import cv2
import numpy as np

def load_images(query_img_path, target_img_path):
    """Load query and target images in grayscale."""
    query_img = cv2.imread(query_img_path, cv2.IMREAD_GRAYSCALE)
    target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
    
    if query_img is None or target_img is None:
        raise ValueError("Error loading images. Check file paths.")
    
    return query_img, target_img

def detect_sift_features(img, sift):
    """Detect SIFT keypoints and descriptors in an image."""
    keypoints, descriptors = sift.detectAndCompute(img, None)
    if descriptors is None:
        raise ValueError("No features detected in the image.")
    return keypoints, descriptors

def match_features(des1, des2, bf, match_ratio=0.1, min_matches=10):
    """Match SIFT features between two images using BFMatcher."""
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    num_matches = max(min_matches, int(len(matches) * match_ratio))
    return matches[:num_matches]

def draw_matches(query_img, kp1, target_img, kp2, matches, output_path):
    """Draw matched keypoints and save the result."""
    result_img = cv2.drawMatches(
        query_img, kp1,
        target_img, kp2,
        matches,
        None,
        matchColor=(0, 255, 0),  # Green lines for matches
        singlePointColor=(0, 0, 255),  # Red dots for keypoints
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(output_path, result_img)
    return result_img

def detect_and_match_object(query_img_path, target_img_path, output_path):
    """Main function to detect and match an object in a target image."""
    # Initialize SIFT and BFMatcher
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    # Load images
    query_img, target_img = load_images(query_img_path, target_img_path)
    
    # Detect features
    kp1, des1 = detect_sift_features(query_img, sift)
    kp2, des2 = detect_sift_features(target_img, sift)
    
    # Match features
    good_matches = match_features(des1, des2, bf)
    
    # Draw and save results
    result_img = draw_matches(query_img, kp1, target_img, kp2, good_matches, output_path)
    
    return result_img

def initialize_video_writer(video_path, output_video_path):
    """Initialize video capture and writer."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file.")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    return cap, out, (width, height)

def compute_homography(kp1, kp2, matches, min_matches=4):
    """Compute homography matrix from matched keypoints."""
    if len(matches) < min_matches:
        return None
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return M

def draw_rectangle_and_points(frame, M, query_img_shape, kp2, matches):
    """Draw bounding rectangle and keypoints on the frame."""
    if M is None:
        return frame
    
    h, w = query_img_shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    
    # Draw rectangle
    frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
    
    # Draw matched keypoints
    for m in matches:
        pt = tuple(np.int32(kp2[m.trainIdx].pt))
        cv2.circle(frame, pt, 3, (0, 0, 255), -1)
    
    return frame

def detect_object_in_video(query_img_path, video_path, output_video_path):
    """Process video to detect object and draw rectangle and keypoints."""
    # Initialize SIFT and BFMatcher
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    # Load query image
    query_img = cv2.imread(query_img_path, cv2.IMREAD_GRAYSCALE)
    if query_img is None:
        raise ValueError("Error loading query image.")
    
    # Detect query image features
    kp1, des1 = detect_sift_features(query_img, sift)
    
    # Initialize video
    cap, out, _ = initialize_video_writer(video_path, output_video_path)
    
    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = detect_sift_features(frame_gray, sift)
        
        good_matches = match_features(des1, des2, bf)
        
        M = compute_homography(kp1, kp2, good_matches)
        frame = draw_rectangle_and_points(frame, M, query_img.shape, kp2, good_matches)
        
        out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # Still image matching
        query_img_path = "images/query.jpg"
        target_img_path = "images/target.jpg"
        output_path = "result.jpg"
        detect_and_match_object(query_img_path, target_img_path, output_path)
        print("Image processing completed. Output saved to:", output_path)
        
        # Video processing
        video_path = "images/car-detection.mp4"
        query_img_path = "images/vid_query.png"
        output_video_path = "output_video.mp4"
        detect_object_in_video(query_img_path, video_path, output_video_path)
        print("Video processing completed. Output saved to:", output_video_path)
    except Exception as e:
        print(f"Error: {str(e)}")
