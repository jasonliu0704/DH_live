import cv2
import numpy as np
import dlib
import os.path
import bz2
from concurrent.futures import ProcessPoolExecutor, wait
from tqdm import tqdm

# Check for actual CUDA support
OPENCV_CUDA_AVAILABLE = False
DLIB_CUDA_AVAILABLE = False

try:
    OPENCV_CUDA_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0
except (cv2.error, AttributeError):
    OPENCV_CUDA_AVAILABLE = False

try:
    DLIB_CUDA_AVAILABLE = dlib.cuda.get_num_devices() > 0
except (AttributeError, Exception):
    DLIB_CUDA_AVAILABLE = False

print(f"OpenCV CUDA support: {'Available' if OPENCV_CUDA_AVAILABLE else 'Not Available'}")
print(f"dlib CUDA support: {'Available' if DLIB_CUDA_AVAILABLE else 'Not Available'}")

# Enable CUDA device selection for dlib if available
if DLIB_CUDA_AVAILABLE:
    def set_dlib_cuda_device(device_id):
        """Set the CUDA device for dlib to use"""
        try:
            # dlib uses a global CUDA device setting
            dlib.cuda.set_device(device_id)
            return True
        except Exception as e:
            print(f"Failed to set dlib CUDA device: {e}")
            return False
else:
    def set_dlib_cuda_device(device_id):
        """Dummy function when CUDA is not available"""
        return False

# Add OpenCV CUDA device selection
def set_opencv_cuda_device(device_id):
    """Set the CUDA device for OpenCV to use"""
    if OPENCV_CUDA_AVAILABLE:
        try:
            cv2.cuda.setDevice(device_id)
            return True
        except Exception as e:
            print(f"Failed to set OpenCV CUDA device: {e}")
            return False
    return False

def cuda_enhanced_clahe(gray_img, clip_limit=2.0, grid_size=(8, 8)):
    """Apply CLAHE using CUDA if available, otherwise use CPU implementation"""
    if OPENCV_CUDA_AVAILABLE:
        try:
            # Upload to GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(gray_img)
            
            # Create CUDA CLAHE
            cuda_clahe = cv2.cuda.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            
            # Apply CLAHE on GPU
            result_gpu = cuda_clahe.apply(gpu_img)
            
            # Download result
            return result_gpu.download()
        except Exception as e:
            print(f"CUDA CLAHE failed, falling back to CPU: {e}")
    
    # CPU fallback
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(gray_img)

def cuda_threshold(img, thresh, maxval, type):
    """Apply threshold using CUDA if available, otherwise use CPU implementation"""
    if OPENCV_CUDA_AVAILABLE:
        try:
            # Upload to GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            
            # Apply threshold on GPU
            _, gpu_result = cv2.cuda.threshold(gpu_img, thresh, maxval, type)
            
            # Download result
            return gpu_result.download()
        except Exception as e:
            print(f"CUDA threshold failed, falling back to CPU: {e}")
    
    # CPU fallback
    _, result = cv2.threshold(img, thresh, maxval, type)
    return result

def cuda_adaptive_threshold(img, maxval, adaptive_method, threshold_type, block_size, C):
    """Apply adaptive threshold using CUDA if available, otherwise use CPU implementation"""
    if OPENCV_CUDA_AVAILABLE:
        try:
            # Upload to GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            
            # Apply adaptive threshold on GPU
            gpu_result = cv2.cuda.adaptiveThreshold(gpu_img, maxval, adaptive_method, threshold_type, block_size, C)
            
            # Download result
            return gpu_result.download()
        except Exception as e:
            print(f"CUDA adaptive threshold failed, falling back to CPU: {e}")
    
    # CPU fallback
    return cv2.adaptiveThreshold(img, maxval, adaptive_method, threshold_type, block_size, C)

def cuda_morphology(img, op, kernel):
    """Apply morphological operations using CUDA if available, otherwise use CPU implementation"""
    if OPENCV_CUDA_AVAILABLE:
        try:
            # Upload to GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            
            # Create CUDA filter
            if op == cv2.MORPH_OPEN:
                gpu_filter = cv2.cuda.createMorphologyFilter(op, cv2.CV_8UC1, kernel)
            elif op == cv2.MORPH_CLOSE:
                gpu_filter = cv2.cuda.createMorphologyFilter(op, cv2.CV_8UC1, kernel)
            else:
                raise ValueError(f"Unsupported morphological operation: {op}")
            
            # Apply filter
            gpu_result = gpu_filter.apply(gpu_img)
            
            # Download result
            return gpu_result.download()
        except Exception as e:
            print(f"CUDA morphology failed, falling back to CPU: {e}")
    
    # CPU fallback
    return cv2.morphologyEx(img, op, kernel)

def detect_teeth(image_path, gpu_id=0):
    """
    Detect teeth in an image with improved accuracy for full teeth coverage
    
    Args:
        image_path: Path to the image file
        gpu_id: GPU device ID to use
        
    Returns:
        teeth_image: Cropped image of teeth region
        teeth_rect: Rectangle coordinates [x, y, width, height]
    """
    print(f"[Worker {gpu_id}] Attempting to detect teeth in {image_path}")
    
    # Set CUDA devices if available
    if DLIB_CUDA_AVAILABLE:
        if set_dlib_cuda_device(gpu_id):
            print(f"[Worker {gpu_id}] Set dlib CUDA device to {gpu_id}")
        else:
            print(f"[Worker {gpu_id}] Failed to set dlib CUDA device")
    
    if OPENCV_CUDA_AVAILABLE:
        if set_opencv_cuda_device(gpu_id):
            print(f"[Worker {gpu_id}] Set OpenCV CUDA device to {gpu_id}")
        else:
            print(f"[Worker {gpu_id}] Failed to set OpenCV CUDA device")
    
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
    
    # Enhance contrast using CLAHE with CUDA if available
    enhanced_gray = cuda_enhanced_clahe(gray, clipLimit=2.0, grid_size=(8, 8))
    
    # Detect faces in enhanced image
    faces = detector(enhanced_gray)
    
    if len(faces) == 0:
        # Try with original grayscale if enhanced fails
        faces = detector(gray)
        if len(faces) == 0:
            print(f"[Worker {gpu_id}] No faces detected in {image_path}")
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
        
        # Try multiple thresholding techniques using CUDA if available
        teeth_regions = []
        
        # 1. Lower thresholds for adaptive detection to catch more subtle teeth
        adaptive_thresh = cuda_adaptive_threshold(
            mouth_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2  # Increased block size, lower constant for subtler detection
        )
        teeth_regions.append(adaptive_thresh)
        
        # 2. Wider range of threshold values including lower values
        for threshold_val in [130, 150, 170]:  # Lower threshold values to detect more subtle teeth
            fixed_thresh = cuda_threshold(mouth_roi, threshold_val, 255, cv2.THRESH_BINARY)
            teeth_regions.append(fixed_thresh)
        
        # 3. Otsu's thresholding
        otsu_thresh = cuda_threshold(mouth_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        teeth_regions.append(otsu_thresh)
        
        # 4. Add TRIANGLE thresholding which can work well for bimodal images
        triangle_thresh = cuda_threshold(mouth_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        teeth_regions.append(triangle_thresh)
        
        # Combine results
        combined_thresh = np.zeros_like(mouth_roi)
        for region in teeth_regions:
            combined_thresh = cv2.bitwise_or(combined_thresh, region)
        
        # Clean up using morphological operations
        kernel = np.ones((3, 3), np.uint8)  # Slightly larger kernel
        opening = cuda_morphology(combined_thresh, cv2.MORPH_OPEN, kernel)
        
        # More aggressive closing to connect teeth regions
        closing = cuda_morphology(opening, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        
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
                print(f"[Worker {gpu_id}] Using fallback region for {image_path}")
    
    # Return None, None if no teeth were detected
    if teeth_image is None:
        print("Warning: Face detected but no teeth found")
    
    return teeth_image, teeth_rect

def process_directory(directory_path, worker_id):
    print(f"[Worker {worker_id}] Starting to process directory: {directory_path}")
    try:
        # Set CUDA devices if available
        if DLIB_CUDA_AVAILABLE:
            if set_dlib_cuda_device(worker_id):
                print(f"[Worker {worker_id}] Set dlib CUDA device to {worker_id}")
            else:
                print(f"[Worker {worker_id}] Failed to set dlib CUDA device")
                
        if OPENCV_CUDA_AVAILABLE:
            if set_opencv_cuda_device(worker_id):
                print(f"[Worker {worker_id}] Set OpenCV CUDA device to {worker_id}")
            else:
                print(f"[Worker {worker_id}] Failed to set OpenCV CUDA device")
        
        image_dir = os.path.join(directory_path, "image")
        output_dir = os.path.join(directory_path, "teeth_seg")
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if image directory exists
        if not os.path.exists(image_dir):
            print(f"[Worker {worker_id}] Error: Image directory {image_dir} does not exist")
            return
            
        if not os.path.isdir(image_dir):
            print(f"[Worker {worker_id}] Error: {image_dir} is not a directory")
            return
            
        # Get list of images
        try:
            image_files = [
                f for f in os.listdir(image_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
        except Exception as e:
            print(f"[Worker {worker_id}] Error listing images in {image_dir}: {str(e)}")
            return
            
        # Now we can safely reference image_files
        print(f"[Worker {worker_id}] Processing directory: {directory_path} with {len(image_files)} images")
        
        if len(image_files) == 0:
            print(f"[Worker {worker_id}] No images found in {image_dir}")
            return
            
        print(f"Processing {len(image_files)} images...")
        teeth_rect_list = []
        
        # Process each image with progress bar
        for image_file in tqdm(image_files, desc=f"Detecting teeth in {os.path.basename(directory_path)}"):
            input_path = os.path.join(image_dir, image_file)
            output_path = os.path.join(output_dir, image_file)
            
            try:
                teeth_img, teeth_coords = detect_teeth(input_path, gpu_id=worker_id)
                
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
        
        print(f"[Worker {worker_id}] Writing detection results to {coords_path}")
        print(f"\nProcessing complete. Results saved in {output_dir}")
        print(f"Coordinates saved to {coords_path}")
    except Exception as e:
        print(f"[Worker {worker_id}] Unexpected error in process_directory: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse, os
    from concurrent.futures import ProcessPoolExecutor

    parser = argparse.ArgumentParser(description="Teeth detection in images using multiple worker processes.")
    parser.add_argument("root_directory", help="Root directory containing subdirectories.")
    parser.add_argument("--workers", type=int, default=2, help="Number of worker processes (default: 2)")
    args = parser.parse_args()

    root_directory = args.root_directory
    worker_count = args.workers

    # Gather subdirectories to process
    subdirectories = [d for d in os.listdir(root_directory) 
                      if os.path.isdir(os.path.join(root_directory, d))]

    # Create worker ID list
    worker_ids = list(range(worker_count))

    # Main block logs
    print(f"Discovered {len(subdirectories)} subdirectories under Root: {root_directory}")
    print(f"Using {worker_count} worker processes")
       
    # Run processing in parallel
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = []
        for i, subdir in enumerate(subdirectories):
            dir_path = os.path.join(root_directory, subdir)
            worker_id = worker_ids[i % len(worker_ids)]
            futures.append(executor.submit(process_directory, dir_path, worker_id))
            print(f"Submitting {subdir} to worker {worker_id}")
        # Wait for all tasks to complete
        wait(futures)
        
        # Check for exceptions in futures
        for future in futures:
            if future.exception():
                print(f"Error in worker: {future.exception()}")
        
        print("All jobs have finished.")