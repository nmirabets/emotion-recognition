import streamlit as st
from emotion_recognition_fastai import EmotionRecognition

# Define emotion recognition model path
model_path = './app/NB_2_vgg16_bn_epochs_3_lr_0.0025.pth'

# Create an instance of the EmotionRecognition class
er = EmotionRecognition(model_path,threshold=0.4)

def main():
    st.image('./app/resources/PulseAI_logo.png', width=250)
    st.caption("Facial Emotion Recognition System - Beta version 1.0.0")
    #st.write("This app captures your webcam feed, identifies faces, and predicts emotions.")

    er.start_video_capture() # Start webcam feed

    if not er.video_capture.isOpened():
        st.error("Error: Webcam not found.")
        return

    # Create an empty placeholder to show the webcam feed
    placeholder = st.empty()

    # Create the stop button
    stop_button = st.button("Stop")

    while not stop_button:
        # Display the frame with detected faces and predicted emotions in the Streamlit app
        #placeholder.image(er.recognize(), channels="BGR")
        placeholder.image(er.recognize(), channels="BGR")

    er.video_capture.release()

if __name__ == "__main__":
    main()
