import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import cvlib as cv
from cvlib.object_detection import draw_bbox

def f_keepLargeComponents(I, th):
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

def detect_objects(frame):
    bbox, labels, conf = cv.detect_common_objects(frame)
    return draw_bbox(frame, bbox, labels, conf)

def process_video(video_path):
    fgModel = cv2.createBackgroundSubtractorMOG2()
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (600, 400))
        fgmask = fgModel.apply(frame)
        num_labels, labels_im = cv2.connectedComponents(np.array(fgmask > 0, np.uint8))
        fgmask = f_keepLargeComponents(labels_im, 1000)
        frame = detect_objects(frame)
        
        F = np.zeros(frame.shape, np.uint8)
        F[:, :, 0], F[:, :, 1], F[:, :, 2] = fgmask, fgmask, fgmask
        F2 = np.hstack((frame, F))
        
        stframe.image(F2, channels="BGR")
        if st.button("Stop Processing"):
            break
    
    cap.release()

def main():
    st.title("Real-Time Object Detection & Motion Tracking")
    
    option = st.radio("Choose an option", ("Live Camera", "Upload Video"))
    
    if option == "Live Camera":
        st.write("Live camera detection is not yet implemented.")
    
    elif option == "Upload Video":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False) as temp_video:
                temp_video.write(uploaded_file.read())
                process_video(temp_video.name)
    
if __name__ == "__main__":
    main()
