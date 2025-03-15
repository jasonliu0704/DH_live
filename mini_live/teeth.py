import cv2
import numpy as np
import dlib
import os.path

def detect_teeth(image_path):
    """
    Detect teeth in an image, crop the region, and return the coordinates
    
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
    faces = detector(gray)
    
    if len(faces) == 0:
        raise ValueError("No faces detected in the image")
    
    teeth_image = None
    teeth_rect = None
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        # Extract mouth region using landmarks
        mouth_points = []
        for i in range(48, 68):  # Landmarks 48-67 represent the mouth region
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            mouth_points.append((x, y))
        
        mouth_points = np.array(mouth_points, dtype=np.int32)
        mouth_rect = cv2.boundingRect(mouth_points)
        mx, my, mw, mh = mouth_rect
        
        # Extract mouth region
        mouth_roi = gray[my:my+mh, mx:mx+mw]
        
        # Apply image processing to detect teeth (white/bright areas in mouth)
        _, thresh = cv2.threshold(mouth_roi, 150, 255, cv2.THRESH_BINARY)
        
        # Clean up using morphological operations
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to focus on teeth
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter by minimum area
                valid_contours.append(contour)
        
        # Combine all detected teeth regions
        if valid_contours:
            # Create mask for all teeth contours
            teeth_mask = np.zeros_like(opening)
            for contour in valid_contours:
                cv2.drawContours(teeth_mask, [contour], -1, 255, -1)
            
            # Find bounding rectangle for all teeth regions
            teeth_contours_combined = np.vstack([cnt for cnt in valid_contours])
            tx, ty, tw, th = cv2.boundingRect(teeth_contours_combined)
            
            # Convert to original image coordinates
            tx += mx
            ty += my
            
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
    import bz2

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