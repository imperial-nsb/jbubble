import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_frame(frame):
    # Convert to grayscale if it's not already
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Invert the image so the bubble ring (dark) becomes bright
    # But wait, the glint is bright. 
    # Let's try to detect the dark ring.
    
    # Simple thresholding might work. Let's try Otsu's or a fixed threshold.
    # The background is around 150-180?
    # Let's use a blurred version to find the bubble.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Try to find the dark ring. Threshold for pixels < threshold_value.
    # We can use binary inversion.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    # Find the largest contour which should be the bubble
    c = max(contours, key=cv2.contourArea)
    
    # Get the minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(c)
    
    return (x, y), radius

def test_on_image(image_path, output_path):
    img = cv2.imread(image_path)
    center, radius = process_frame(img)
    
    if center:
        # Draw the circle
        cv2.circle(img, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 1)
        # Draw the center
        cv2.circle(img, (int(center[0]), int(center[1])), 1, (0, 0, 255), -1)
        
    cv2.imwrite(output_path, img)
    return radius

radiusA = test_on_image('/Users/crt25/code/cattaneo_radius_extract/bubbleA_first_frame.jpg', '/Users/crt25/code/cattaneo_radius_extract/test_A.jpg')
radiusB = test_on_image('/Users/crt25/code/cattaneo_radius_extract/bubbleB_first_frame.jpg', '/Users/crt25/code/cattaneo_radius_extract/test_B.jpg')

print(f"Radius A: {radiusA}")
print(f"Radius B: {radiusB}")
