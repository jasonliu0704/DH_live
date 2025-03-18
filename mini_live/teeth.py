import cv2
import numpy as np
import dlib
import os.path
import bz2
from concurrent.futures import ProcessPoolExecutor, wait


# Add CUDA support for dlib
if dlib.DLIB_USE_CUDA:
    print("Using CUDA for dlib")
else:
    print("CUDA not available for dlib")

def detect_teeth(image_path, gpu_id=0):
    """
    Detect teeth in an image with improved accuracy for full teeth coverage
    
    Args:
        image_path: Path to the image file
        
    Returns:
        teeth_image: Cropped image of teeth region
        teeth_rect: Rectangle coordinates [x, y, width, height]
    """
    # Additional logging in detect_teeth
    print(f"[GPU {gpu_id}] Attempting to detect teeth in {image_path}")
    
    # Set CUDA device
    if dlib.DLIB_USE_CUDA:
        cv2.cuda.setDevice(gpu_id)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Check for shape predictor file
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    if not os.path.exists(predictor_path):
        import urllib.request
        compressed_path = predictor_path + '.bz2'
        if not os.path.exists(compressed_path):
            url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
            print(f"Downloading shape predictor from {url} ...")
            urllib.request.urlretrieve(url, compressed_path)
        print("Decompressing predictor file...")
        with bz2.open(compressed_path, 'rb') as f_in, open(predictor_path, 'wb') as f_out:
            f_out.write(f_in.read())
    if not os.path.exists(predictor_path):
        raise FileNotFoundError(f"Required file not found: {predictor_path}. Please download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    
    # Initialize dlib's face detector and facial landmarks predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    
    # Detect faces in enhanced image
    faces = detector(enhanced_gray)
    
    if len(faces) == 0:
        # Try with original grayscale if enhanced fails
        faces = detector(gray)
        if len(faces) == 0:
            print(f"[GPU {gpu_id}] No faces detected in {image_path}")
            raise ValueError("No faces detected in the image")
    
    teeth_image = None
    teeth_rect = None
    
    for face in faces:
        # Get face bounding box coordinates
        face_x = face.left()
        face_y = face.top()
        face_w = face.width()
        face_h = face.height()
        
        landmarks = predictor(gray, face)
        
        # Extract mouth region using landmarks with minimal padding
        mouth_points = []
        for i in range(48, 68):  # Landmarks 48-67 represent the mouth region
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            mouth_points.append((x, y))
        
        mouth_points = np.array(mouth_points, dtype=np.int32)
        mouth_rect = cv2.boundingRect(mouth_points)
        mx, my, mw, mh = mouth_rect
        
        # Add minimal padding to mouth region (10% instead of 25%)
        padding_x = int(mw * 0.1)
        padding_y = int(mh * 0.1)
        
        # Ensure padded coordinates stay within image bounds
        mx_padded = max(0, mx - padding_x)
        my_padded = max(0, my - padding_y)
        mw_padded = min(image.shape[1] - mx_padded, mw + 2 * padding_x)
        mh_padded = min(image.shape[0] - my_padded, mh + 2 * padding_y)
        
        # Extract mouth region with padding
        mouth_roi = enhanced_gray[my_padded:my_padded+mh_padded, mx_padded:mx_padded+mw_padded]
        
        # Try multiple thresholding techniques
        teeth_regions = []
        
        # 1. Lower thresholds for adaptive detection to catch more subtle teeth
        adaptive_thresh = cv2.adaptiveThreshold(
            mouth_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2  # Increased block size, lower constant for subtler detection
        )
        teeth_regions.append(adaptive_thresh)
        
        # 2. Wider range of threshold values including lower values
        for threshold_val in [130, 150, 170]:  # Lower threshold values to detect more subtle teeth
            _, fixed_thresh = cv2.threshold(mouth_roi, threshold_val, 255, cv2.THRESH_BINARY)
            teeth_regions.append(fixed_thresh)
        
        # 3. Otsu's thresholding
        _, otsu_thresh = cv2.threshold(mouth_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        teeth_regions.append(otsu_thresh)
        
        # 4. Add TRIANGLE thresholding which can work well for bimodal images
        _, triangle_thresh = cv2.threshold(mouth_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        teeth_regions.append(triangle_thresh)
        
        # Combine results
        combined_thresh = np.zeros_like(mouth_roi)
        for region in teeth_regions:
            combined_thresh = cv2.bitwise_or(combined_thresh, region)
        
        # Clean up using morphological operations
        kernel = np.ones((3, 3), np.uint8)  # Slightly larger kernel
        opening = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel)
        
        # More aggressive closing to connect teeth regions
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        
        # Find contours
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Less selective contour filtering
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h > 0 else 0
            
            # More relaxed filtering (smaller area, wider aspect ratio)
            if area > 20 and 0.1 < aspect_ratio < 4.0:  # Lower area threshold, wider aspect ratio
                # Check average brightness of the region (less strict)
                mask = np.zeros_like(mouth_roi)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mean_brightness = cv2.mean(mouth_roi, mask=mask)[0]
                
                # More relaxed brightness requirement
                if mean_brightness > np.mean(mouth_roi) + 10:  # Lower brightness difference threshold
                    valid_contours.append(contour)
        
        # Save debug image to help diagnose issues
        debug_image = cv2.cvtColor(mouth_roi, cv2.COLOR_GRAY2BGR)
        for contour in valid_contours:
            cv2.drawContours(debug_image, [contour], -1, (0, 255, 0), 1)
        debug_path = f"debug_mouth_{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
        cv2.imwrite(debug_path, debug_image)
        
        # Combine all detected teeth regions
        if valid_contours:
            # Create mask for all teeth contours
            teeth_mask = np.zeros_like(closing)
            for contour in valid_contours:
                cv2.drawContours(teeth_mask, [contour], -1, 255, -1)
            
            # Find bounding rectangle for all teeth regions
            teeth_contours_combined = np.vstack([cnt for cnt in valid_contours])
            tx, ty, tw, th = cv2.boundingRect(teeth_contours_combined)
            
            # Add minimal padding to teeth region (5% instead of 20%)
            teeth_padding_x = int(tw * 0.05)
            teeth_padding_y = int(th * 0.05)
            
            # Convert to original image coordinates with minimal padding
            tx = mx_padded + max(0, tx - teeth_padding_x)
            ty = my_padded + max(0, ty - teeth_padding_y)
            tw = min(image.shape[1] - tx, tw + 2 * teeth_padding_x)
            th = min(image.shape[0] - ty, th + 2 * teeth_padding_y)
            
            # Ensure teeth region stays within face boundaries
            tx = max(tx, face_x)
            ty = max(ty, face_y)
            tw = min(tw, face_x + face_w - tx)
            th = min(th, face_y + face_h - ty)
            
            # Validate that region dimensions are positive
            if tw <= 0 or th <= 0:
                print("Warning: Detected teeth region has invalid dimensions after face boundary check")
                continue
            
            # Crop teeth region from original image
            teeth_image = image[ty:ty+th, tx:tx+tw]
            teeth_rect = [tx, ty, tw, th]
            
            # Validate if the region is likely to contain teeth
            # Simple validation: check if region is in the lower half of the face
            face_midpoint_y = face_y + face_h / 2
            if ty < face_midpoint_y:
                if ty + th > face_midpoint_y:  # At least partially in lower face
                    break
                else:
                    print("Warning: Detected teeth region is in upper half of face, might be incorrect")
                    # We'll still use it but continue to check other faces
            else:
                break  # Good detection, in lower half of face
        else:
            # If no teeth contours found, try a fallback approach - use the inner region of the lower face
            # Focus on the lower third of the face, middle 60% horizontally
            inner_x = face_x + int(face_w * 0.2)
            inner_y = face_y + int(face_h * 0.6)  # Lower 40% of face
            inner_w = int(face_w * 0.6)
            inner_h = int(face_h * 0.25)  # About 25% of face height
            
            # Ensure coordinates are valid
            inner_x = max(0, inner_x)
            inner_y = max(0, inner_y)
            inner_w = min(inner_w, image.shape[1] - inner_x)
            inner_h = min(inner_h, image.shape[0] - inner_y)
            
            # Only use this fallback if it's a valid region
            if inner_w > 10 and inner_h > 5:
                teeth_image = image[inner_y:inner_y+inner_h, inner_x:inner_x+inner_w]
                teeth_rect = [inner_x, inner_y, inner_w, inner_h]
                print(f"[GPU {gpu_id}] Using fallback region for {image_path}")
    
    # Return None, None if no teeth were detected
    if teeth_image is None:
        print("Warning: Face detected but no teeth found")
    
    return teeth_image, teeth_rect

def process_directory(directory_path, gpu_id):
    # Set GPU if available
    if dlib.DLIB_USE_CUDA:
        cv2.cuda.setDevice(gpu_id)
    
    # Additional logging in process_directory
    print(f"[GPU {gpu_id}] Processing directory: {directory_path} with {len(image_files)} images")
    
    # Create output directory
    image_dir = os.path.join(directory_path, "image")
    output_dir = os.path.join(directory_path, "teeth_seg")
    os.makedirs(output_dir, exist_ok=True)
    # Get list of images
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Processing {len(image_files)} images...")
    teeth_rect_list = []
    
    # Process each image with progress bar
    for image_file in tqdm(image_files, desc=f"Detecting teeth in {os.path.basename(directory_path)}"):
        input_path = os.path.join(image_dir, image_file)
        output_path = os.path.join(output_dir, image_file)
        
        try:
            teeth_img, teeth_coords = detect_teeth(input_path, gpu_id=gpu_id)
            
            if teeth_img is not None:
                # Save teeth image
                cv2.imwrite(output_path, teeth_img)
                teeth_rect_list.append(teeth_coords)
            else:
                # Create empty image with same size as input
                original_img = cv2.imread(input_path)
                empty_img = np.zeros_like(original_img)
                cv2.imwrite(output_path, empty_img)
                teeth_rect_list.append([0, 0, 0, 0])
                
        except Exception as e:
            print(f"\nError processing {image_file}: {e}")
            # Create empty image for failed detection
            original_img = cv2.imread(input_path)
            empty_img = np.zeros_like(original_img)
            cv2.imwrite(output_path, empty_img)
            teeth_rect_list.append([0, 0, 0, 0])
    
    # Save all teeth coordinates
    coords_path = os.path.join(output_dir, "all.txt")
    np.savetxt(coords_path, np.array(teeth_rect_list), fmt='%d')
    
    print(f"[GPU {gpu_id}] Writing detection results to {coords_path}")
    print(f"\nProcessing complete. Results saved in {output_dir}")
    print(f"Coordinates saved to {coords_path}")

if __name__ == "__main__":
    import argparse, os
    from concurrent.futures import ProcessPoolExecutor

    parser = argparse.ArgumentParser(description="Teeth detection in images using specified GPUs.")
    parser.add_argument("root_directory", help="Root directory containing subdirectories.")
    parser.add_argument("--gpus", type=int, default=2, help="Number of GPUs available (default: 2)")
    args = parser.parse_args()

    root_directory = args.root_directory
    gpu_count = args.gpus

    # Gather subdirectories to process
    subdirectories = [d for d in os.listdir(root_directory) 
                      if os.path.isdir(os.path.join(root_directory, d))]

    # Create GPU list based on the user-specified number
    gpus = list(range(gpu_count))

    # Main block logs
    print(f"Discovered {len(subdirectories)} subdirectories under Root: {root_directory}")
    print(f"Using {gpu_count} GPUs: {gpus}")
    for i, subdir in enumerate(subdirectories):
        print(f"Submitting {subdir} to GPU {gpus[i % gpu_count]}")
    
    # Run processing in parallel using the available GPUs
    with ProcessPoolExecutor(max_workers=gpu_count) as executor:
        futures = []
        for i, subdir in enumerate(subdirectories):
            dir_path = os.path.join(root_directory, subdir)
            futures.append(executor.submit(process_directory, dir_path, gpus[i % len(gpus)]))
        # Wait for all tasks to complete
        wait(futures)
        print("All jobs have finished.")