import cv2
import os
import numpy as np
# import tkinter as tk
# from tkinter import filedialog, simpledialog
import torch
import torchvision

def get_video_details(video_path):
    if not os.path.exists(video_path):
        print(f"Error: The file '{video_path}' was not found.")
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open the video file '{video_path}'.")
        return None
    file_stat = os.stat(video_path)
    file_size = file_stat.st_size
    file_type = os.path.splitext(video_path)[1]
    video_name = os.path.basename(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_seconds = frame_count / fps if fps > 0 else 0
    details = {
        "Video Name": video_name,
        "File Path": os.path.abspath(video_path),
        "File Size": f"{file_size / (1024 * 1024):.2f} MB",
        "File Type": file_type.upper(),
        "Resolution": f"{width}x{height}",
        "Frames Per Second (FPS)": f"{fps:.2f}",
        "Frame Count": frame_count,
        "Duration": f"{duration_seconds:.2f} seconds",
        "Author": "Not Available via OpenCV"
    }
    cap.release()
    return details

# Load DeepLabV3 model
def load_deeplab_model():
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
    return model

def extract_and_remove_person_frames(video_path, interval_seconds, output_folder="extracted_frames_bgonly"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open the video.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Error: Could not determine FPS. Cannot extract frames.")
        return
    frame_interval = int(interval_seconds * fps)
    if frame_interval <= 0:
        print("Error: Invalid interval. Please enter a positive number.")
        return
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    frame_no = 0
    extracted_count = 0

    model = load_deeplab_model()
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    print(f"\nExtracting and removing person every {interval_seconds} seconds...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no % frame_interval == 0:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = preprocess(rgb_frame).unsqueeze(0)

                with torch.no_grad():
                    output = model(input_tensor)["out"][0]
                mask = (output.argmax(0).byte().cpu().numpy() == 15)

                # Dilate mask to increase the edge around the detected person
                kernel_size = 15  # Increase this value for a larger border cropping
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

                bgra_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                bgra_frame[dilated_mask, 3] = 0

                image_filename = os.path.join(output_folder, f"frame_bgonly_{int(frame_no)}.png")
                cv2.imwrite(image_filename, bgra_frame)
                extracted_count += 1
                print(f"Saved background with transparent person at {timestamp:.2f}s (Frame No: {frame_no}) as '{image_filename}'")
            except Exception as e:
                print(f"Error processing frame {frame_no}: {e}. Skipping this frame.")

        frame_no += 1
    cap.release()
    print(f"\nExtraction and person removal complete! Total of {extracted_count} frames saved to '{output_folder}'.")
    
def main():
    print("🎬 Video Frame Background Extractor (Person Transparent)")
    try:
        root = tk.Tk()
        root.withdraw()
        root.deiconify()
        video_path = filedialog.askopenfilename(
            title="Select a Video File",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv *.wmv"),
                ("All files", ".")
            ]
        )
        if not video_path:
            print("No file selected. Exiting.")
            return
        print("\n" + "="*50)
        print("🔍 Video Details")
        print("="*50)
        details = get_video_details(video_path)
        if details:
            for key, value in details.items():
                print(f"🔹 {key}: {value}")
        print("="*50 + "\n")
        try:
            interval = simpledialog.askfloat(
                "Frame Extraction",
                "Enter the interval in seconds to extract frames (e.g., 5.0):"
            )
            if interval is None:
                print("Operation cancelled. Exiting.")
                return
            if interval <= 0:
                raise ValueError
        except ValueError:
            print("Error: Invalid input. Please enter a positive number.")
            return
    finally:
        if 'root' in locals():
            root.destroy()

    extract_and_remove_person_frames(video_path, interval)

if __name__ == "__main__":
    main()
