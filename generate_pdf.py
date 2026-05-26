import os
import sys
import glob
import uuid
import shutil
# import tkinter as tk
# from tkinter import filedialog, simpledialog, messagebox
from PIL import Image

# Import project modules (assumes they sit alongside this file)
# slide_change_detector.py must expose: detect_slide_changes(video_path, min_slide_duration_sec, sample_interval_sec) -> (intervals, fps)
# frame_ext_removal.py must expose: extract_and_remove_person_frames(video_path, interval_seconds, output_folder)
# create_composite_image.py must expose: create_composite_image(input_folder, output_path)
import slide_change_detector
import frame_ext_removal
import create_composite_image


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def frames_for_interval(video_path, start_frame, end_frame, fps, step_seconds, out_dir):
    """
    Extract frames between [start_frame, end_frame] every step_seconds and write raw BGR PNGs.
    Uses OpenCV internal reader via frame_ext_removal to stay consistent with environment.
    """
    import cv2  # local import to avoid extra dependency at module import time
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video for raw extraction.")
        return 0
    frame_interval = max(1, int(step_seconds * fps))
    saved = 0
    f = start_frame
    while f <= end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(out_dir, f"raw_{f}.png")
        cv2.imwrite(out_path, frame)
        saved += 1
        f += frame_interval
    cap.release()
    return saved


def run_pipeline():
    root = tk.Tk()
    root.withdraw()

    # 1) Pick video
    video_path = filedialog.askopenfilename(
        title="Select a Lecture Video",
        filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.wmv"), ("All files", ".*")]
    )
    if not video_path:
        print("No video selected. Exiting.")
        return

    # 2) Ask parameters
    min_slide_duration_sec = simpledialog.askfloat(
        "Minimum Slide Duration (seconds)",
        "Enter the minimum slide duration in seconds (e.g., 15 or 30):",
        minvalue=1.0,
        initialvalue=20.0
    )
    if min_slide_duration_sec is None:
        print("Cancelled.")
        return

    sample_interval_sec = simpledialog.askfloat(
        "Slide Sampling Interval (seconds)",
        "Sampling interval for slide-change detection (e.g., 2.0):",
        minvalue=0.5,
        initialvalue=2.0
    )
    if sample_interval_sec is None:
        print("Cancelled.")
        return

    per_slide_frame_step_sec = simpledialog.askfloat(
        "Per-Slide Frame Step (seconds)",
        "During background building, sample frames every N seconds within each slide (e.g., 1.5):",
        minvalue=0.2,
        initialvalue=1.5
    )
    if per_slide_frame_step_sec is None:
        print("Cancelled.")
        return

    person_dilate_kernel_hint = simpledialog.askinteger(
        "Person Mask Dilation",
        "Approx. dilation kernel size for person removal (e.g., 15). Leave blank to use default in your module.",
        minvalue=3
    )
    # This value is only informational; the actual dilation kernel is inside frame_ext_removal module.
    # Kept here in case the internal function is later adapted to read from env or a param.

    # 3) Detect slide intervals
    print("\n=== Detecting slide changes ===")
    intervals, fps = slide_change_detector.detect_slide_changes(
        video_path,
        min_slide_duration_sec=min_slide_duration_sec,
        sample_interval_sec=sample_interval_sec
    )
    if not intervals:
        messagebox.showwarning("No Slides Found", "No slide intervals were detected with the current parameters.")
        return

    # 4) Workspace
    session_id = uuid.uuid4().hex[:8]
    work_root = os.path.join(os.path.dirname(video_path), f"distill_session_{session_id}")
    ensure_dir(work_root)

    # 5) For each slide interval, sample raw frames and then run person-removal on those sampled frames.
    # Because frame_ext_removal.extract_and_remove_person_frames() operates on time-interval sampling,
    # and we need per-interval control, we’ll:
    #   - Extract our own raw frames per interval at per_slide_frame_step_sec.
    #   - For each saved raw frame, run person-removal by calling the model once per image
    #     using a small helper that mimics the internal logic.
    print("\n=== Preparing DeepLab model (one-time load) ===")
    import cv2
    import torch
    import numpy as np
    import torchvision

    # Load model once
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
    preprocess = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def remove_person_make_transparent(bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess(rgb).unsqueeze(0)
        with torch.no_grad():
            out = model(input_tensor)["out"][0]
        mask = (out.argmax(0).byte().cpu().numpy() == 15)
        # Dilate
        kernel_size = person_dilate_kernel_hint if person_dilate_kernel_hint else 15
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        bgra = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2BGRA)
        bgra[dilated, 3] = 0
        return bgra

    composite_dir = os.path.join(work_root, "composites")
    ensure_dir(composite_dir)

    final_slide_images = []

    print("\n=== Building background per slide ===")
    for idx, (start_f, end_f) in enumerate(intervals, start=1):
        slide_dir = os.path.join(work_root, f"slide_{idx:03d}")
        raw_dir = os.path.join(slide_dir, "raw")
        masked_dir = os.path.join(slide_dir, "masked")
        ensure_dir(raw_dir)
        ensure_dir(masked_dir)

        # Extract raw frames for this interval
        saved = frames_for_interval(
            video_path=video_path,
            start_frame=start_f,
            end_frame=end_f,
            fps=fps,
            step_seconds=per_slide_frame_step_sec,
            out_dir=raw_dir
        )
        print(f"Slide {idx}: saved {saved} raw frames from interval [{start_f}, {end_f}].")

        # Person removal on saved frames
        raw_frames = sorted(glob.glob(os.path.join(raw_dir, "*.png")))
        for rf in raw_frames:
            bgr = cv2.imread(rf, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            bgra = remove_person_make_transparent(bgr)
            out_name = os.path.join(masked_dir, os.path.basename(rf).replace("raw_", "masked_"))
            cv2.imwrite(out_name, bgra)

        # Composite to single clean slide
        composite_path = os.path.join(composite_dir, f"slide_{idx:03d}.png")
        # Use the provided function to union-fill transparent regions
        create_composite_image.create_composite_image(masked_dir, composite_path)
        if os.path.exists(composite_path):
            final_slide_images.append(composite_path)

    if not final_slide_images:
        messagebox.showwarning("No Slides", "No composite slides were created.")
        return

    # 6) Generate PDF
    print("\n=== Generating PDF ===")
    pdf_name = simpledialog.askstring(
        "Output PDF Name",
        "Enter output PDF filename (e.g., Distill.pdf):",
        initialvalue="Distill.pdf"
    )
    if not pdf_name:
        pdf_name = "Distill.pdf"

    output_pdf_path = os.path.join(os.path.dirname(video_path), pdf_name)

    # Load images via PIL and save to a single PDF
    pil_images = []
    for path in final_slide_images:
        img = Image.open(path).convert("RGB")
        pil_images.append(img)

    if pil_images:
        first = pil_images[0]
        rest = pil_images[1:] if len(pil_images) > 1 else []
        first.save(output_pdf_path, save_all=True, append_images=rest)
        print(f"PDF saved at: {output_pdf_path}")
        messagebox.showinfo("Success", f"PDF created:\n{output_pdf_path}")
    else:
        messagebox.showwarning("No Images", "No images available to write into PDF.")

    # Optional: Keep workspace or ask to clean
    if messagebox.askyesno("Cleanup", "Delete intermediate working folders?"):
        try:
            shutil.rmtree(work_root)
            print("Cleaned intermediate data.")
        except Exception as e:
            print(f"Cleanup error: {e}")


if __name__ == "__main__":
    run_pipeline()
