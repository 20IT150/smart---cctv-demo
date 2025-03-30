import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import time
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="Motion Detection and Object Tracking",
    page_icon="🎥",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-bottom: 1rem;
        color: #34495e;
    }
    .stAlert {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 0.25rem;
    }
    .thumbnail-container {
        display: flex;
        overflow-x: auto;
        padding: 10px 0;
    }
    .thumbnail {
        width: 120px;
        height: 80px;
        margin-right: 10px;
        border: 2px solid transparent;
    }
</style>
""", unsafe_allow_html=True)

# Utility functions from backend
def f_keepLargeComponents(I, th):
    """Keep only large connected components in a binary image"""
    R = np.zeros(I.shape) < 0
    unique_labels = np.unique(I.flatten())
    for label in unique_labels:
        if label == 0:
            pass
        else:
            I2 = I == label
            if np.sum(I2) > th:
                R = R | I2
    return np.float32(255 * R)

def convert_to_image(cv2_img):
    """Convert OpenCV image to PIL Image for Streamlit display"""
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def get_image_base64(img):
    """Convert image to base64 for HTML display"""
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def process_frame(frame, bg_subtractor, min_area):
    """Process a frame with background subtraction and component filtering"""
    # Resize for consistent processing
    frame = cv2.resize(frame, dsize=(600, 400))
    
    # Apply background subtraction
    fgmask = bg_subtractor.apply(frame)
    
    # Apply morphological operations
    K_r = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    fgmask = cv2.morphologyEx(np.float32(fgmask), cv2.MORPH_OPEN, K_r)
    
    # Connected components analysis
    num_labels, labels_im = cv2.connectedComponents(np.array(fgmask > 0, np.uint8))
    
    # Keep only large components
    fgmask = f_keepLargeComponents(labels_im, min_area)
    
    # Create visualization mask
    F = np.zeros(frame.shape, np.uint8)
    F[:, :, 0], F[:, :, 1], F[:, :, 2] = fgmask, fgmask, fgmask
    
    # Combine original and mask
    combined = np.hstack((frame, F))
    
    has_motion = np.sum(fgmask) > 0
    
    return frame, fgmask, combined, has_motion

# App title and description
st.markdown("<div class='main-header'>Motion Detection and Object Tracking</div>", unsafe_allow_html=True)
st.markdown("Track objects and detect motion in videos or image sequences. Upload your own video or use your webcam.")

# Sidebar for settings
st.sidebar.markdown("<div class='sub-header'>Settings</div>", unsafe_allow_html=True)

# Input source selection
input_source = st.sidebar.radio("Select Input Source", ["Upload Video", "Upload Image Sequence", "Sample Video"])

# Motion detection parameters
min_area = st.sidebar.slider("Minimum Component Area", 100, 5000, 1000, 100)
history = st.sidebar.slider("Background History", 100, 1000, 500, 50)
var_threshold = st.sidebar.slider("Variance Threshold", 5, 100, 16, 1)
detect_shadows = st.sidebar.checkbox("Detect Shadows", value=True)

# Output settings
save_output = st.sidebar.checkbox("Save Processed Frames", value=False)
min_sequence_frames = st.sidebar.number_input("Minimum Frames in Sequence", 1, 20, 5)

# Object detection option
use_object_detection = st.sidebar.checkbox("Enable Object Detection", value=False)
if use_object_detection:
    st.sidebar.warning("Note: Object detection requires the cvlib package which must be installed on your Streamlit Cloud deployment.")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Process Video", "Results", "About"])

with tab1:
    # Process input based on selection
    if input_source == "Upload Video":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
        
        if uploaded_file is not None:
            # Save uploaded file to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(uploaded_file.read())
            
            # Create background subtractor
            fgModel = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold, detectShadows=detect_shadows)
            
            # Process video
            if st.button("Process Video"):
                # Create output directory if saving results
                if save_output:
                    output_dir = tempfile.mkdtemp()
                    st.session_state['output_dir'] = output_dir
                    st.session_state['saved_frames'] = []
                
                # Open video capture
                cap = cv2.VideoCapture(temp_file.name)
                
                # Process frames
                frame_idx = 0
                motion_sequence = []
                sequence_counter = 0
                
                progress_bar = st.progress(0)
                frame_display = st.empty()
                status_text = st.empty()
                
                # Get total frames for progress calculation
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_idx += 1
                    
                    # Process frame
                    original, mask, combined, has_motion = process_frame(frame, fgModel, min_area)
                    
                    # Update motion sequence
                    if has_motion:
                        motion_sequence.append(original)
                    elif len(motion_sequence) > 0:
                        # Save sequence if it meets minimum length
                        if len(motion_sequence) >= min_sequence_frames and save_output:
                            sequence_counter += 1
                            for i, seq_frame in enumerate(motion_sequence):
                                frame_name = f"{sequence_counter}_{i+1}.jpg"
                                frame_path = os.path.join(output_dir, frame_name)
                                
                                # Apply object detection if enabled
                                if use_object_detection:
                                    try:
                                        import cvlib as cv
                                        from cvlib.object_detection import draw_bbox
                                        bbox, labels, conf = cv.detect_common_objects(seq_frame)
                                        seq_frame = draw_bbox(seq_frame, bbox, labels, conf)
                                    except ImportError:
                                        st.warning("cvlib not available. Skipping object detection.")
                                
                                cv2.imwrite(frame_path, seq_frame)
                                st.session_state['saved_frames'].append(frame_path)
                        
                        # Reset sequence
                        motion_sequence = []
                    
                    # Display current frame
                    frame_display.image(convert_to_image(combined), caption="Processing: Original | Foreground Mask", use_column_width=True)
                    
                    # Update progress
                    progress = min(frame_idx / total_frames, 1.0)
                    progress_bar.progress(progress)
                    
                    # Display status
                    status_text.text(f"Processing frame {frame_idx}/{total_frames} | Motion sequences: {sequence_counter}")
                    
                    # Slow down processing slightly for display
                    time.sleep(0.01)
                
                # Save final sequence if any
                if len(motion_sequence) >= min_sequence_frames and save_output:
                    sequence_counter += 1
                    for i, seq_frame in enumerate(motion_sequence):
                        frame_name = f"{sequence_counter}_{i+1}.jpg"
                        frame_path = os.path.join(output_dir, frame_name)
                        
                        # Apply object detection if enabled
                        if use_object_detection:
                            try:
                                import cvlib as cv
                                from cvlib.object_detection import draw_bbox
                                bbox, labels, conf = cv.detect_common_objects(seq_frame)
                                seq_frame = draw_bbox(seq_frame, bbox, labels, conf)
                            except ImportError:
                                pass
                        
                        cv2.imwrite(frame_path, seq_frame)
                        st.session_state['saved_frames'].append(frame_path)
                
                cap.release()
                
                # Complete
                progress_bar.progress(1.0)
                status_text.success(f"Processing complete! {sequence_counter} motion sequences detected.")
                
                # Clean up
                os.unlink(temp_file.name)
                
                # Set results flag
                if save_output and sequence_counter > 0:
                    st.session_state['has_results'] = True
                    st.info("Results are available in the Results tab.")
    
    elif input_source == "Upload Image Sequence":
        uploaded_files = st.file_uploader("Upload image sequence", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        if uploaded_files:
            # Create background subtractor
            fgModel = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold, detectShadows=detect_shadows)
            
            # Process images
            if st.button("Process Images"):
                # Create output directory if saving results
                if save_output:
                    output_dir = tempfile.mkdtemp()
                    st.session_state['output_dir'] = output_dir
                    st.session_state['saved_frames'] = []
                
                # Process frames
                frame_idx = 0
                motion_sequence = []
                sequence_counter = 0
                
                progress_bar = st.progress(0)
                frame_display = st.empty()
                status_text = st.empty()
                
                total_frames = len(uploaded_files)
                
                for file in uploaded_files:
                    frame_idx += 1
                    
                    # Read image
                    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    # Reset file pointer
                    file.seek(0)
                    
                    # Process frame
                    original, mask, combined, has_motion = process_frame(frame, fgModel, min_area)
                    
                    # Update motion sequence
                    if has_motion:
                        motion_sequence.append(original)
                    elif len(motion_sequence) > 0:
                        # Save sequence if it meets minimum length
                        if len(motion_sequence) >= min_sequence_frames and save_output:
                            sequence_counter += 1
                            for i, seq_frame in enumerate(motion_sequence):
                                frame_name = f"{sequence_counter}_{i+1}.jpg"
                                frame_path = os.path.join(output_dir, frame_name)
                                
                                # Apply object detection if enabled
                                if use_object_detection:
                                    try:
                                        import cvlib as cv
                                        from cvlib.object_detection import draw_bbox
                                        bbox, labels, conf = cv.detect_common_objects(seq_frame)
                                        seq_frame = draw_bbox(seq_frame, bbox, labels, conf)
                                    except ImportError:
                                        st.warning("cvlib not available. Skipping object detection.")
                                
                                cv2.imwrite(frame_path, seq_frame)
                                st.session_state['saved_frames'].append(frame_path)
                        
                        # Reset sequence
                        motion_sequence = []
                    
                    # Display current frame
                    frame_display.image(convert_to_image(combined), caption="Processing: Original | Foreground Mask", use_column_width=True)
                    
                    # Update progress
                    progress = min(frame_idx / total_frames, 1.0)
                    progress_bar.progress(progress)
                    
                    # Display status
                    status_text.text(f"Processing frame {frame_idx}/{total_frames} | Motion sequences: {sequence_counter}")
                    
                    # Slow down processing slightly for display
                    time.sleep(0.01)
                
                # Save final sequence if any
                if len(motion_sequence) >= min_sequence_frames and save_output:
                    sequence_counter += 1
                    for i, seq_frame in enumerate(motion_sequence):
                        frame_name = f"{sequence_counter}_{i+1}.jpg"
                        frame_path = os.path.join(output_dir, frame_name)
                        
                        # Apply object detection if enabled
                        if use_object_detection:
                            try:
                                import cvlib as cv
                                from cvlib.object_detection import draw_bbox
                                bbox, labels, conf = cv.detect_common_objects(seq_frame)
                                seq_frame = draw_bbox(seq_frame, bbox, labels, conf)
                            except ImportError:
                                pass
                        
                        cv2.imwrite(frame_path, seq_frame)
                        st.session_state['saved_frames'].append(frame_path)
                
                # Complete
                progress_bar.progress(1.0)
                status_text.success(f"Processing complete! {sequence_counter} motion sequences detected.")
                
                # Set results flag
                if save_output and sequence_counter > 0:
                    st.session_state['has_results'] = True
                    st.info("Results are available in the Results tab.")
    
    else:  # Sample video
        st.info("Using a sample video for demonstration")
        
        # Create sample video data - in a real app, you'd use a sample video file
        sample_video = st.selectbox("Select sample video", ["Campus", "Shopping Mall", "Office Room"])
        
        # Process sample video
        if st.button("Process Sample Video"):
            # Create background subtractor
            fgModel = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold, detectShadows=detect_shadows)
            
            # Create output directory if saving results
            if save_output:
                output_dir = tempfile.mkdtemp()
                st.session_state['output_dir'] = output_dir
                st.session_state['saved_frames'] = []
            
            # Generate some dummy frames for demonstration
            total_frames = 100
            progress_bar = st.progress(0)
            frame_display = st.empty()
            status_text = st.empty()
            
            sequence_counter = 0
            motion_sequence = []
            
            # Simulate processing frames
            for i in range(total_frames):
                # Create a dummy frame with some motion
                frame = np.zeros((400, 600, 3), dtype=np.uint8)
                
                # Add some moving objects
                if i % 10 < 5:  # Motion every other 5 frames
                    cv2.circle(frame, (300 + i % 100, 200), 50, (0, 0, 255), -1)
                    has_motion = True
                else:
                    has_motion = False
                
                # Apply background subtraction (simplified for demo)
                if i == 0:
                    fgmask = np.zeros((400, 600), dtype=np.float32)
                else:
                    # Process frame - simplified for demo
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    fgmask = fgModel.apply(frame)
                    fgmask = np.float32(fgmask)
                
                # Create visualization mask
                F = np.zeros(frame.shape, np.uint8)
                F[:, :, 0], F[:, :, 1], F[:, :, 2] = fgmask, fgmask, fgmask
                
                # Combine original and mask
                combined = np.hstack((frame, F))
                
                # Update motion sequence
                if has_motion:
                    motion_sequence.append(frame)
                elif len(motion_sequence) > 0:
                    # Save sequence if it meets minimum length
                    if len(motion_sequence) >= min_sequence_frames and save_output:
                        sequence_counter += 1
                        for j, seq_frame in enumerate(motion_sequence):
                            frame_name = f"{sequence_counter}_{j+1}.jpg"
                            frame_path = os.path.join(output_dir, frame_name)
                            cv2.imwrite(frame_path, seq_frame)
                            st.session_state['saved_frames'].append(frame_path)
                    
                    # Reset sequence
                    motion_sequence = []
                
                # Display current frame
                frame_display.image(convert_to_image(combined), caption="Processing: Original | Foreground Mask", use_column_width=True)
                
                # Update progress
                progress = (i + 1) / total_frames
                progress_bar.progress(progress)
                
                # Display status
                status_text.text(f"Processing frame {i+1}/{total_frames} | Motion sequences: {sequence_counter}")
                
                # Slow down processing slightly for display
                time.sleep(0.05)
            
            # Save final sequence if any
            if len(motion_sequence) >= min_sequence_frames and save_output:
                sequence_counter += 1
                for j, seq_frame in enumerate(motion_sequence):
                    frame_name = f"{sequence_counter}_{j+1}.jpg"
                    frame_path = os.path.join(output_dir, frame_name)
                    cv2.imwrite(frame_path, seq_frame)
                    st.session_state['saved_frames'].append(frame_path)
            
            # Complete
            progress_bar.progress(1.0)
            status_text.success(f"Processing complete! {sequence_counter} motion sequences detected.")
            
            # Set results flag
            if save_output and sequence_counter > 0:
                st.session_state['has_results'] = True
                st.info("Results are available in the Results tab.")

with tab2:
    st.markdown("<div class='sub-header'>Detected Motion Sequences</div>", unsafe_allow_html=True)
    
    if 'has_results' in st.session_state and st.session_state['has_results']:
        if 'saved_frames' in st.session_state and st.session_state['saved_frames']:
            # Group frames by sequence
            sequences = {}
            for frame_path in st.session_state['saved_frames']:
                frame_name = os.path.basename(frame_path)
                seq_id = frame_name.split('_')[0]
                
                if seq_id not in sequences:
                    sequences[seq_id] = []
                
                sequences[seq_id].append(frame_path)
            
            # Display sequences
            selected_sequence = st.selectbox("Select motion sequence", list(sequences.keys()))
            
            if selected_sequence:
                st.write(f"Sequence {selected_sequence} - {len(sequences[selected_sequence])} frames")
                
                # Display thumbnails
                cols = st.columns(min(5, len(sequences[selected_sequence])))
                for i, (col, frame_path) in enumerate(zip(cols, sequences[selected_sequence])):
                    img = Image.open(frame_path)
                    col.image(img, caption=f"Frame {i+1}", use_column_width=True)
                
                # Display full sequence
                st.write("Full sequence:")
                sequence_frames = []
                for frame_path in sequences[selected_sequence]:
                    img = Image.open(frame_path)
                    sequence_frames.append(np.array(img))
                
                # Create animated GIF option
                if st.button("Create GIF from Sequence"):
                    with st.spinner("Creating GIF..."):
                        # Create a temporary file for the GIF
                        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as temp_gif:
                            gif_path = temp_gif.name
                        
                        # Convert frames to GIF
                        images = [Image.fromarray(frame) for frame in sequence_frames]
                        images[0].save(
                            gif_path,
                            save_all=True,
                            append_images=images[1:],
                            duration=200,
                            loop=0
                        )
                        
                        # Display the GIF
                        with open(gif_path, 'rb') as gif_file:
                            gif_data = gif_file.read()
                        
                        st.image(gif_data, caption="Motion Sequence GIF")
                        
                        # Provide download link
                        st.download_button(
                            label="Download GIF",
                            data=gif_data,
                            file_name=f"sequence_{selected_sequence}.gif",
                            mime="image/gif"
                        )
                        
                        # Clean up
                        os.unlink(gif_path)
                
                # Display stacked view
                st.write("All frames in sequence:")
                for i, frame in enumerate(sequence_frames):
                    st.image(frame, caption=f"Frame {i+1}", use_column_width=True)
    else:
        st.info("No results available yet. Process a video or image sequence with 'Save Processed Frames' enabled.")

with tab3:
    st.markdown("<div class='sub-header'>About this application</div>", unsafe_allow_html=True)
    
    st.markdown("""
    This application implements motion detection and object tracking using background subtraction and connected component analysis.
    
    ### Features:
    - Process videos or image sequences to detect motion
    - Background subtraction using the MOG2 algorithm
    - Connected component filtering to remove noise
    - Optional object detection (requires cvlib)
    - Save and display motion sequences
    - Export sequences as GIFs
    
    ### How to use:
    1. Select an input source (upload video, image sequence, or use sample)
    2. Adjust parameters in the sidebar
    3. Process the input
    4. View results in the Results tab
    
    ### Requirements for deployment:
    ```
    streamlit
    opencv-python-headless
    numpy
    pillow
    matplotlib
    ```
    
    For object detection functionality, also install:
    ```
    cvlib
    tensorflow
    ```
    
    ### How it works:
    The application uses background subtraction to identify moving objects in a video sequence. It then applies connected component analysis to filter out small noise components and track larger moving objects. Optionally, it can apply object detection to identify the types of objects detected.
    """)
    
    # Show parameters explanation
    st.markdown("### Parameter Explanation:")
    
    st.markdown("""
    - **Minimum Component Area**: The minimum size (in pixels) of connected components to keep. Smaller components are filtered out as noise.
    - **Background History**: Number of frames used to build the background model in MOG2.
    - **Variance Threshold**: Threshold on the squared Mahalanobis distance to decide whether a pixel is foreground or background.
    - **Detect Shadows**: Enable shadow detection in MOG2 algorithm.
    - **Minimum Frames in Sequence**: The minimum number of consecutive frames with motion to save as a sequence.
    """)
    
    # Add deployment instructions
    st.markdown("### Deployment to Streamlit Cloud:")
    
    st.markdown("""
    To deploy this application to Streamlit Cloud:
    
    1. Create a GitHub repository with this code
    2. Include a `requirements.txt` file with the necessary dependencies
    3. Connect your GitHub repository to Streamlit Cloud
    4. Deploy the application
    
    Example `requirements.txt`:
    ```
    streamlit==1.24.0
    opencv-python-headless==4.7.0.72
    numpy==1.24.3
    pillow==9.5.0
    matplotlib==3.7.1
    ```
    """)
