import streamlit as st
import cv2
import os
import tempfile
import glob
import torch
import torchvision
import numpy as np
from PIL import Image
import PIL.JpegImagePlugin

# Import your local modules 
# Make sure tkinter is removed from slide_change_detector.py and create_composite_image.py!
import slide_change_detector
import create_composite_image

# --- PAGE SETUP ---
st.set_page_config(page_title="LectureLens AI", page_icon="🎓")
st.title("🎓 LectureLens AI Demo")
st.write("Upload a short video lecture to extract clean, unobstructed PDF notes.")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Processing Settings")
min_slide_duration = st.sidebar.slider(
    "Min Slide Duration (sec)", 
    min_value=1.0, max_value=60.0, value=20.0, step=1.0,
    help="Increase if you get duplicate slides."
)
sample_interval = st.sidebar.slider(
    "Slide Sample Interval (sec)", 
    min_value=0.5, max_value=5.0, value=2.0, step=0.5,
    help="Interval for slide-change detection."
)
per_slide_frame_step = st.sidebar.slider(
    "Per-Slide Frame Step (sec)", 
    min_value=0.2, max_value=5.0, value=1.5, step=0.1,
    help="Sample frames every N seconds within each slide to build the background."
)
person_dilate_kernel = st.sidebar.slider(
    "Person Mask Dilation", 
    min_value=3, max_value=31, value=15, step=2,
    help="Increase this for a larger border cropping around the presenter."
)

# --- CACHE AI MODEL ---
# This ensures the 100MB+ model is only loaded into memory once, not every time a user clicks a button.
@st.cache_resource
def load_ai_model():
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
    preprocess = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    return model, preprocess

model, preprocess = load_ai_model()

# --- CORE LOGIC FUNCTIONS ---
def remove_person_make_transparent(bgr_img):
    """Runs DeepLabV3 to find the person and make those pixels transparent."""
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess(rgb).unsqueeze(0)
    with torch.no_grad():
        out = model(input_tensor)["out"][0]
    mask = (out.argmax(0).byte().cpu().numpy() == 15)
    
    # Dilate mask to ensure edges of the person are fully removed
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (person_dilate_kernel, person_dilate_kernel))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    
    # Convert to BGRA (Alpha channel added) and set person pixels to transparent (0)
    bgra = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2BGRA)
    bgra[dilated, 3] = 0
    return bgra

def frames_for_interval(video_path, start_frame, end_frame, fps, step_seconds, out_dir):
    """Extracts raw frames from a specific time interval in the video."""
    if not os.path.exists(out_dir): 
        os.makedirs(out_dir)
    
    cap = cv2.VideoCapture(video_path)
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


# --- MAIN UI WORKFLOW ---

# 1. Initialize session state to hold the final PDF bytes so it survives page reloads
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # 2. File Size Validation
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > 50:
        st.error(f"File is {size_mb:.1f}MB. Please keep prototype videos under 50MB.")
    else:
        # Save upload to a temp file because OpenCV requires a physical file path
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
        tfile.write(uploaded_file.read())
        
        # 3. Video Duration Validation
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = frame_count / fps if fps > 0 else 0
        cap.release()

        if duration_sec > 600:
            st.error(f"Video is {duration_sec/60:.1f} minutes long. Limit is 10 minutes.")
        else:
            st.success(f"Video accepted! (Size: {size_mb:.1f}MB, Duration: {duration_sec/60:.1f} min)")
            
            # 4. Trigger Processing
            if st.button("Generate Notes"):
                # Create a persistent output directory so files aren't deleted
                output_dir = os.path.join(os.getcwd(), "output")
                os.makedirs(output_dir, exist_ok=True)
                
                with tempfile.TemporaryDirectory() as work_root:
                    try:
                        st.text("Step 1: Detecting slide transitions...")
                        intervals, fps = slide_change_detector.detect_slide_changes(
                            tfile.name, min_slide_duration, sample_interval
                        )
                        
                        if not intervals:
                            st.warning("No slide changes detected with current settings.")
                        else:
                            st.write(f"✅ Found {len(intervals)} unique slides.")
                            
                            # Save final composites in the persistent output directory
                            composite_dir = os.path.join(output_dir, "composites")
                            os.makedirs(composite_dir, exist_ok=True)
                            final_slide_images = []
                            
                            progress_bar = st.progress(0)
                            
                            for idx, (start_f, end_f) in enumerate(intervals, start=1):
                                st.text(f"Processing Slide {idx}/{len(intervals)}...")
                                slide_dir = os.path.join(work_root, f"slide_{idx:03d}")
                                raw_dir = os.path.join(slide_dir, "raw")
                                masked_dir = os.path.join(slide_dir, "masked")
                                os.makedirs(raw_dir, exist_ok=True)
                                os.makedirs(masked_dir, exist_ok=True)

                                # Extract frames for this slide
                                frames_for_interval(tfile.name, start_f, end_f, fps, per_slide_frame_step, raw_dir)
                                raw_frames = sorted(glob.glob(os.path.join(raw_dir, "*.png")))
                                
                                # Run AI Erasing on each frame
                                for rf in raw_frames:
                                    bgr = cv2.imread(rf, cv2.IMREAD_COLOR)
                                    if bgr is not None:
                                        bgra = remove_person_make_transparent(bgr)
                                        out_name = os.path.join(masked_dir, os.path.basename(rf).replace("raw_", "masked_"))
                                        cv2.imwrite(out_name, bgra)

                                # Stitch the background
                                composite_path = os.path.join(composite_dir, f"slide_{idx:03d}.png")
                                create_composite_image.create_composite_image(masked_dir, composite_path)
                                
                                if os.path.exists(composite_path):
                                    final_slide_images.append(composite_path)
                                    
                                progress_bar.progress(idx / len(intervals))

                            # 5. PDF Generation (Handling Transparency correctly)
                            st.text("Step 3: Compiling PDF...")
                            pdf_path = os.path.join(output_dir, "Lecture_Notes.pdf")
                            
                            pil_images = []
                            for p in final_slide_images:
                                # Open the transparent PNG
                                img = Image.open(p).convert("RGBA")
                                # Create a solid white background
                                background = Image.new("RGB", img.size, (255, 255, 255))
                                # Paste the transparent image onto the white background
                                background.paste(img, mask=img.split()[3]) 
                                pil_images.append(background)
                            
                            if pil_images:
                                # Save out the PDF
                                pil_images[0].save(pdf_path, save_all=True, append_images=pil_images[1:])
                                
                                # Save PDF to session state so it survives the download button refresh
                                with open(pdf_path, "rb") as pdf_file:
                                    st.session_state.pdf_bytes = pdf_file.read()
                                    
                                st.success("🎉 Notes successfully generated! Click the button below to download.")
                                
                    except Exception as e:
                        st.error(f"An error occurred during processing: {e}")
                        
            # 6. Display Download Button (Placed OUTSIDE the generation block)
            if st.session_state.pdf_bytes is not None:
                st.download_button(
                    label="⬇️ Download Lecture Notes (PDF)", 
                    data=st.session_state.pdf_bytes, 
                    file_name="LectureLens_Notes.pdf",
                    mime="application/pdf"
                )