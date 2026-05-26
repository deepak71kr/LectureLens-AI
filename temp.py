import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog
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


def frame_to_gray(frame, size=(256, 256)):
    """Convert frame to grayscale and resize for robustness."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, size)
    return resized


def compute_ssim(frame1, frame2):
    """Compute SSIM similarity (0-1)."""
    return ssim(frame1, frame2)


def seconds_to_min_sec(seconds):
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    return f"{minutes} min {sec} sec"


def get_frame(cap, idx, frame_count):
    """Fetch frame safely at index."""
    if idx < 0 or idx >= frame_count:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame_to_gray(frame)


def verify_continuity(cap, candidate_index, fps, frame_count, offset=10, threshold=0.85):
    """
    Check continuity:
    - Compare candidate frame with k+offset and k-offset.
    - If candidate is similar to either, it's a false change.
    - Otherwise, it's a real slide change.
    """
    candidate = get_frame(cap, candidate_index, frame_count)
    before = get_frame(cap, candidate_index - offset, frame_count)
    after = get_frame(cap, candidate_index + offset, frame_count)

    if candidate is None:
        return False

    # Compare with before and after
    if before is not None:
        score_before = compute_ssim(candidate, before)
        if score_before > threshold:
            return False  # false change (same as before)

    if after is not None:
        score_after = compute_ssim(candidate, after)
        if score_after > threshold:
            return False  # false change (same as after)

    return True  # confirmed slide change


def detect_slide_changes(video_path,
                         min_slide_duration_sec=30,
                         sample_interval_sec=2,
                         ssim_threshold=0.75,
                         continuity_offset=10):
    """
    Detect slide changes with continuity verification.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open the video file '{video_path}'.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_interval_frames = int(sample_interval_sec * fps)
    min_slide_interval_frames = int(min_slide_duration_sec * fps)

    sampled_frames = []
    sampled_indices = []

    # Sample frames
    for i in range(0, frame_count, sample_interval_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        sampled_frames.append(frame_to_gray(frame))
        sampled_indices.append(i)

    slide_change_indices = [0]  # start frame

    for i in range(1, len(sampled_frames)):
        score = compute_ssim(sampled_frames[i - 1], sampled_frames[i])
        if score < ssim_threshold:  # candidate change
            candidate_index = sampled_indices[i]
            if candidate_index - slide_change_indices[-1] >= min_slide_interval_frames:
                # Verify continuity
                if verify_continuity(cap, candidate_index, fps, frame_count,
                                     offset=continuity_offset, threshold=0.85):
                    slide_change_indices.append(candidate_index)

    slide_change_indices.append(frame_count - 1)  # end frame
    cap.release()

    intervals = []
    for i in range(len(slide_change_indices) - 1):
        start = slide_change_indices[i]
        end = slide_change_indices[i + 1] - 1
        intervals.append([start, end])

    return intervals, fps


def main():
    print("📑 Continuity-Based Slide Change Detector")

    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.wmv"), ("All files", ".*")]
    )
    if not video_path:
        print("No file selected. Exiting.")
        return

    fps, frame_count = get_video_details(video_path)
    if fps is None:
        return

    print(f"Video FPS: {fps:.2f}, Total Frames: {frame_count}")

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

    print(f"\nUsing minimum slide duration = {min_slide_duration_sec} sec and sample interval = {sample_interval_sec} sec.")


if __name__ == "__main__":
    main()
