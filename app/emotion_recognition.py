import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import cv2
import face_recognition

def detect_faces(image):
    # Load the image and convert it from BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Find face locations in the image
    face_locations = face_recognition.face_locations(rgb_image)

    for (top, right, bottom, left) in face_locations:
        # Draw a rectangle around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

    return image, face_locations  # Return both the modified image and face locations

class EmotionRecognition:

    resolution = (640, 480)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def __init__(self, model_path, threshold=0.5, ):
        # Load your emotion recognition model
        self.model = torch.load(model_path)

        self.device = torch.device("cpu") # Use CPU for inference
        self.model.to(self.device)  # Move model to CPU
        self.model.eval() # Set model to evaluation mode

        self.threshold = threshold # Emotion probability threshold

    def start_video_capture(self, video_capture=cv2.VideoCapture(0)):

        # Create a VideoCapture object
        self.video_capture = video_capture

        # Set the webcam resolution
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])


    def predict_emotion(self, image):
        # Preprocess the image
        new_transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        # Make predictions
        with torch.no_grad():
            image = Image.fromarray(image)  # Convert NumPy array to PIL Image
            image = new_transform(image).unsqueeze(0).to(self.device) # Preprocess the image and unsqueeze to add batch dimension
            output = self.model(image) # Make predictions
            _, predicted = torch.max(output.data, 1) # Get the predicted class with highest probability

            probabilities = F.softmax(output, dim=1) 
            emotion_probability = torch.round(probabilities * 1000) / 1000  # Round to 2 decimal places

            predicted_emotion = self.emotion_labels[predicted.item()]
            return predicted_emotion, emotion_probability[0][predicted.item()].item()
        
    def recognize(self):

        ret, frame = self.video_capture.read() # Read the frame

        if ret: # If the frame was properly read.

            # Convert the frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces in the frame
            frame_with_faces, face_locations = detect_faces(frame_rgb)

            for (top, right, bottom, left) in face_locations:
                # Crop the face region
                face_image = frame_with_faces[top:bottom, left:right]

                # Predict the emotion
                predicted_emotion, emotion_probability = self.predict_emotion(face_image)

                # Only display the emotion label and its probability if it's above the threshold
                if emotion_probability > self.threshold:
                    label = f"{predicted_emotion} {emotion_probability*100:.0f}%"

                    cv2.putText(frame_with_faces, label, (left, bottom + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                
            return cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB)

