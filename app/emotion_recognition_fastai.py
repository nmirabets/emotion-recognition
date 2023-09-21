from fastai.vision.all import *
import cv2

def detect_faces(image):
    # Initialize the Haar Cascade face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale (Haar cascades work better in grayscale)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find face locations in the image
    face_locations = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in face_locations:
        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 1)

    # Return the image and face locations in the format [(top, right, bottom, left)]
    formatted_face_locations = [(y, x+w, y+h, x) for (x, y, w, h) in face_locations]
    return image, formatted_face_locations

class EmotionRecognition:
    
    resolution = (640, 480)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def __init__(self, model_path, threshold=0.5):
        # Load your emotion recognition model
        self.learn = load_learner(model_path)

        self.threshold = threshold # Emotion probability threshold

    def start_video_capture(self, video_capture=cv2.VideoCapture(0)):

        # Create a VideoCapture object
        self.video_capture = video_capture

        # Set the webcam resolution
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

    def predict_emotion(self, image):

        # Convert NumPy array to PIL Image and use fastai's PILImage to ensure it's in the right format
        img = PILImage.create(Image.fromarray(image))
        
        # Resize and convert the image to grayscale
        img = img.resize((48, 48))
        img = img.convert("L")
        
        # Convert back to fastai's PILImage
        img = PILImage(img)
        
        # Make predictions
        predicted, predicted_index, probs = self.learn.predict(img)
        predicted_emotion = str(predicted)
        emotion_probability = probs[predicted_index].item()

        return predicted_emotion, emotion_probability
    
    def load_image_from_path(self, image_path):
        """
        Load an image from the given path.
        """
        self.image = cv2.imread(image_path)

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
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 225, 225), 1, cv2.LINE_AA)

            return cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB)
        
    def recognize_loaded_img(self):

        self.load_image_from_path('/Users/nmirabets/Documents/data-science/repos/emotion-recognition/home_gathering.png')

        if self.image is None:
            raise ValueError("No image loaded. Please load an image using load_image_from_path method.")

        frame = self.image

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
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 230, 230), 1, cv2.LINE_AA)

        return cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB)