#!/usr/bin/env python3
"""Quick test with timeout for Yoosee RTSP."""

import cv2
import sys
import threading
import time

result = {"success": False, "frame": None}

def test_camera(url):
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            result["success"] = True
            result["frame"] = frame
        cap.release()
    return result["success"]

if __name__ == "__main__":
    ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.100.46"
    user = sys.argv[2] if len(sys.argv) > 2 else "admin"
    password = sys.argv[3] if len(sys.argv) > 3 else "123"
    
    url = f"rtsp://{user}:{password}@{ip}:554/onvif1"
    print(f"Testing: rtsp://{user}:****@{ip}:554/onvif1")
    
    # Test in thread with timeout
    t = threading.Thread(target=test_camera, args=(url,))
    t.start()
    t.join(timeout=10)
    
    if result["success"]:
        print(f"SUCCESS! Frame shape: {result['frame'].shape}")
    else:
        print("FAILED - Could not connect")
