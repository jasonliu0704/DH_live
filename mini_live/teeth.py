import cv2
import numpy as np
import dlib
import os.path
import bz2

def detect_teeth(image_path):
    """
    Detect teeth in an image with improved accuracy for full teeth coverage
    
    Args:
        image_path: Path to the image file
        
    Returns:
        teeth_image: Cropped image of teeth region
        teeth_rect: Rectangle coordinates [x, y, width, height]
    """
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
            raise ValueError("No faces detected in the image")
    
    teeth_image = None
    teeth_rect = None
    
    for face in faces:
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
        
        # 1. More aggressive adaptive threshold for tighter detection
        adaptive_thresh = cv2.adaptiveThreshold(
            mouth_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 7, 3  # Smaller block size, higher constant
        )
        teeth_regions.append(adaptive_thresh)
        
        # 2. Higher threshold values to focus on the brightest areas (teeth)
        for threshold_val in [150, 170, 190]:  # Higher threshold values
            _, fixed_thresh = cv2.threshold(mouth_roi, threshold_val, 255, cv2.THRESH_BINARY)
            teeth_regions.append(fixed_thresh)
        
        # 3. Otsu's thresholding
        _, otsu_thresh = cv2.threshold(mouth_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        teeth_regions.append(otsu_thresh)
        
        # Combine results
        combined_thresh = np.zeros_like(mouth_roi)
        for region in teeth_regions:
            combined_thresh = cv2.bitwise_or(combined_thresh, region)
        
        # Clean up using morphological operations
        kernel = np.ones((2, 2), np.uint8)  # Smaller kernel for finer detail
        opening = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel)
        
        # More selective closing to avoid merging teeth with non-teeth regions
        small_closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(small_closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # More selective contour filtering
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h > 0 else 0
            
            # Filter by area, aspect ratio and brightness
            if area > 30 and 0.2 < aspect_ratio < 3.0:
                # Check average brightness of the region
                mask = np.zeros_like(mouth_roi)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mean_brightness = cv2.mean(mouth_roi, mask=mask)[0]
                
                # Only include regions that are significantly brighter
                if mean_brightness > np.mean(mouth_roi) + 20:
                    valid_contours.append(contour)
        
        # Combine all detected teeth regions
        if valid_contours:
            # Create mask for all teeth contours
            teeth_mask = np.zeros_like(small_closing)
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
            
            # Crop teeth region from original image
            teeth_image = image[ty:ty+th, tx:tx+tw]
            teeth_rect = [tx, ty, tw, th]
            break
    
    # Return None, None if no teeth were detected
    if teeth_image is None:
        print("Warning: Face detected but no teeth found")
    
    return teeth_image, teeth_rect

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python teeth.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    try:
        teeth_img, teeth_coords = detect_teeth(image_path)
        
        if teeth_img is not None:
            print(f"Teeth detected at coordinates: {teeth_coords}")
            # cv2.imshow('Detected Teeth', teeth_img)
            output_path = f"teeth_{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
            cv2.imwrite(output_path, teeth_img)
            print(f"Saved teeth image to {output_path}")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No teeth detected in the image")
    except Exception as e:
        print(f"Error: {e}")