import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def process_video(video_path, output_csv, frame_rate=10e6):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {video_path} ({frame_count} frames)...")

    results = []
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Otsu thresholding to find the bubble ring
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(c)
        else:
            x, y, radius = np.nan, np.nan, np.nan
            
        time = i / frame_rate
        results.append({
            'frame': i,
            'time_s': time,
            'center_x': x,
            'center_y': y,
            'radius_px': radius
        })

    cap.release()
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")
    return df

# Main execution
if __name__ == "__main__":
    frame_rate = 10e6  # 10 MHz from the paper
    
    dfA = process_video('bubbleA.mp4', 'radius_curve_A.csv', frame_rate)
    dfB = process_video('bubbleB.mp4', 'radius_curve_B.csv', frame_rate)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    if dfA is not None:
        plt.plot(dfA['time_s'] * 1e6, dfA['radius_px'], label='Bubble A', color='blue')
    
    if dfB is not None:
        plt.plot(dfB['time_s'] * 1e6, dfB['radius_px'], label='Bubble B', color='red')
        
    plt.xlabel('Time (µs)')
    plt.ylabel('Radius (pixels)')
    plt.title('Microbubble Radius-Time Curves')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig('radius_time_curves.png', dpi=300)
    print("Saved plot to radius_time_curves.png")
    plt.show() # This might not work in some environments, but we have the file.
