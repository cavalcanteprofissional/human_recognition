#!/usr/bin/env python3
"""
Gateway RTSP to HTTP using FFmpeg.
This allows OpenCV to access the RTSP stream via HTTP.
"""

import subprocess
import threading
import time
import sys
import os
from pathlib import Path

def start_rtsp_gateway(rtsp_url, http_port=8554):
    """
    Start FFmpeg to convert RTSP to HTTP MJPEG stream.
    
    Args:
        rtsp_url: RTSP URL of the camera
        http_port: Port for HTTP stream
    
    Returns:
        Process object
    """
    http_url = f"http://localhost:{http_port}/stream"
    
    print(f"Starting RTSP to HTTP gateway...")
    print(f"  RTSP: {rtsp_url}")
    print(f"  HTTP: {http_url}")
    
    # FFmpeg command to convert RTSP to HTTP MJPEG
    cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-q:v", "5",
        "-f", "mjpeg",
        "-http_port", str(http_port),
        "http://localhost:8554/stream"
    ]
    
    # Try different FFmpeg versions
    commands_to_try = [
        cmd,
        # With verbose output
        ["ffmpeg", "-loglevel", "debug", "-rtsp_transport", "tcp", 
         "-i", rtsp_url, "-vcodec", "mjpeg", "-f", "mjpeg", 
         f"http://localhost:{http_port}/stream"],
    ]
    
    for cmd in commands_to_try:
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL
            )
            time.sleep(2)
            
            if process.poll() is None:
                print(f"FFmpeg started successfully!")
                return process
        except Exception as e:
            print(f"Failed: {e}")
            continue
    
    return None


def check_ffmpeg():
    """Check if FFmpeg is available."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False


def main():
    if not check_ffmpeg():
        print("FFmpeg not found!")
        print("\nPlease install FFmpeg:")
        print("  Windows: winget install ffmpeg")
        print("  Or download from: https://ffmpeg.org/download.html")
        return
    
    # Test URL
    rtsp_url = "rtsp://admin:123@192.168.100.46:554/onvif1"
    
    print("Testing FFmpeg RTSP to HTTP conversion...")
    process = start_rtsp_gateway(rtsp_url)
    
    if process:
        print("\nGateway running!")
        print("Press Ctrl+C to stop")
        try:
            process.wait()
        except KeyboardInterrupt:
            process.terminate()
            print("\nGateway stopped")
    else:
        print("Failed to start gateway")


if __name__ == "__main__":
    main()
