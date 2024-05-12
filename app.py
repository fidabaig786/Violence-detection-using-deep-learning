import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Load pre-trained model
model = load_model('vgg_violence_model.h5')

# Initialize the image mean for mean subtraction
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")

# Initialize the predictions queue
Q = []

# Data augmentation for preprocessing
data_augmentation = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

def preprocess_frame(frame):
    # Apply data augmentation
    augmented_image = data_augmentation.random_transform(frame)
    
    # Resize frame to match model input shape
    resized_frame = cv2.resize(augmented_image, (224, 224))
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Perform mean subtraction
    frame = rgb_frame.astype("float32")
    frame -= mean
    
    return frame

def main():
    st.title("Video Violence Detection")

    st.image("image.png", use_column_width=True)
        
    # Upload video file
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "mov"])

    if uploaded_file:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        
        # Display uploaded video
        st.video("temp_video.mp4")

        # Add a button to trigger violence detection
        if st.sidebar.button("Detect Violence"):
            # Open video file
            cap = cv2.VideoCapture("temp_video.mp4")

            # Get dimensions of the video frames
            ret, frame = cap.read()
            if ret:
                H, W, _ = frame.shape

                # Loop over frames from the video file stream
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Preprocess frame
                    preprocessed_frame = preprocess_frame(frame)

                    # Make predictions on the frame
                    preds = model.predict(np.expand_dims(preprocessed_frame, axis=0))[0]
                    Q.append(preds)

                    # Perform prediction averaging
                    results = np.array(Q).mean(axis=0)
                    violence_prob = results[1]

                    # Print intermediate results
                    print("Violence probability:", violence_prob)

                # Decide label based on violence probability
                overall_label = 'Violence Detected' if violence_prob > 0.15 else 'No Violence Detected'

                # Print the result
                st.write(overall_label)

            # Release the file pointers
            cap.release()
            cv2.destroyAllWindows()

            # Remove temporary file
            os.remove("temp_video.mp4")

if __name__ == "__main__":
    main()
