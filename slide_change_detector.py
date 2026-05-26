import cv2
import os
import numpy as np
# import tkinter as tk
# from tkinter import filedialog, simpledialog
from skimage.metrics import structural_similarity as ssim


def get_video_details(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open the video file '{video_path}'.")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, frame_count


def frame_to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def compute_frame_difference(frame1, frame2):
    gray1 = frame_to_gray(frame1)
    gray2 = frame_to_gray(frame2)
    score, _ = ssim(gray1, gray2, full=True)
    return score


def seconds_to_min_sec(seconds):
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    return f"{minutes} min {sec} sec"


def detect_slide_changes(video_path, min_slide_duration_sec=30, sample_interval_sec=2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open the video file '{video_path}'.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_interval_frames = int(sample_interval_sec * fps)
    min_slide_interval_frames = int(min_slide_duration_sec * fps)

    sampled_frames = []
    sampled_frame_indices = []

    # Sample frames every sample_interval_frames
    for i in range(0, frame_count, sample_interval_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        sampled_frames.append(frame)
        sampled_frame_indices.append(i)

    cap.release()

    slide_change_indices = [0]  # start from frame 0
    threshold_ssim = 0.75  # can be adjusted

    for i in range(1, len(sampled_frames)):
        score = compute_frame_difference(sampled_frames[i - 1], sampled_frames[i])
        if score < threshold_ssim:
            if sampled_frame_indices[i] - slide_change_indices[-1] >= min_slide_interval_frames:
                slide_change_indices.append(sampled_frame_indices[i])

    slide_change_indices.append(frame_count - 1)  # end frame

    intervals = []
    for i in range(len(slide_change_indices) - 1):
        start = slide_change_indices[i]
        end = slide_change_indices[i + 1] - 1
        intervals.append([start, end])

    return intervals, fps


def main():
    print("🖼️ Slide Change Detector with User-Input Minimum Duration")

    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.wmv"), ("All files", ".")]
    )
    if not video_path:
        print("No file selected. Exiting.")
        return

    fps, frame_count = get_video_details(video_path)
    if fps is None:
        return

    print(f"Video FPS: {fps:.2f}, Total Frames: {frame_count}")

    # Ask user for minimum slide duration in seconds
    while True:
        min_slide_duration_sec = simpledialog.askfloat(
            "Minimum Slide Duration",
            "Enter the minimum slide duration in seconds (e.g., 5 or 30):",
            minvalue=1.0
        )
        if min_slide_duration_sec is None:
            print("Operation cancelled. Exiting.")
            return
        if min_slide_duration_sec > 0:
            break
        print("Please enter a positive number.")

    sample_interval_sec = 2

    intervals, fps = detect_slide_changes(
        video_path,
        min_slide_duration_sec=min_slide_duration_sec,
        sample_interval_sec=sample_interval_sec
    )

    print("\nDetected slide intervals (frame ranges and time ranges):")
    for start, end in intervals:
        start_sec = start / fps
        end_sec = end / fps
        print(f"[{start}, {end}] => [{seconds_to_min_sec(start_sec)}, {seconds_to_min_sec(end_sec)}]")

    print(f"\nUsing minimum slide duration = {min_slide_duration_sec} seconds and sample interval = {sample_interval_sec} seconds.")


if __name__ == "__main__":
    main()
