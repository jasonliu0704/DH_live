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
                print("Using fallback face-based region for teeth detection")
    
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
            output_path = f"teeth_{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
            cv2.imwrite(output_path, teeth_img)
            print(f"Saved teeth image to {output_path}")
            
            # Save debug visualization of the detected region on original image with face rectangle
            debug_img = cv2.imread(image_path)
            if debug_img is not None and teeth_coords:
                # Draw face rectangle
                detector = dlib.get_frontal_face_detector()
                faces = detector(cv2.cvtColor(debug_img, cv2.COLOR_BGR2GRAY))
                for face in faces:
                    face_x = face.left()
                    face_y = face.top()
                    face_w = face.width()
                    face_h = face.height()
                    cv2.rectangle(debug_img, (face_x, face_y), (face_x+face_w, face_y+face_h), (255, 0, 0), 2)
                
                # Draw teeth rectangle
                x, y, w, h = teeth_coords
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                debug_output = f"debug_{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
                cv2.imwrite(debug_output, debug_img)
                print(f"Saved debug image to {debug_output}")
            
            # cv2.imshow('Detected Teeth', teeth_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No teeth detected in the image")
            # Save debug visualization showing the face detection
            debug_img = cv2.imread(image_path)
            if debug_img is not None:
                gray = cv2.cvtColor(debug_img, cv2.COLOR_BGR2GRAY)
                detector = dlib.get_frontal_face_detector()
                predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
                faces = detector(gray)
                
                for face in faces:
                    # Draw face rectangle
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Draw mouth landmarks
                    landmarks = predictor(gray, face)
                    for i in range(48, 68):  # Mouth landmarks
                        x = landmarks.part(i).x
                        y = landmarks.part(i).y
                        cv2.circle(debug_img, (x, y), 2, (0, 0, 255), -1)
                
                debug_output = f"debug_noTeeth_{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
                cv2.imwrite(debug_output, debug_img)
                print(f"Saved debug image to {debug_output}")
    except Exception as e:
        print(f"Error: {e}")