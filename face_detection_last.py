import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
loaded_model = keras.models.load_model('my_model.keras')

# Load pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Streamlit camera input
# st.title("Face Detection App")
st.markdown(f"<h1 style='text-align: center;'>Mask classification App</h1>", unsafe_allow_html=True)

st.write("Take a picture to detect faces.")

img_file_buffer = st.camera_input("Take a picture")
classes = ['WithMask','WithoutMask']

if img_file_buffer is not None:
    # Convert the image buffer to OpenCV format
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Convert to grayscale (Haar Cascade works on grayscale images)
    gray_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=4)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(cv2_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        amount = w // 3
        text_position = (x + amount, y - 10)  # Position above the rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
        font_scale = 1  # Font size
        font_color = (0, 255, 0)  # Text color (green)
        font_thickness = 2  # Text thickness
        title = "Without Mask"
        cv2.putText(cv2_img, title, text_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    cropped_face = cv2_img
    test_image = cv2.resize(cropped_face,(128,128))
    test_image = np.reshape(test_image,[1,128,128,3])
    prediction = loaded_model.predict(test_image)
    results = classes[np.argmax(prediction)]
    # Display the processed image
    st.image(cv2_img, channels="BGR", caption=f"Detected {len(faces)} face(s)")

    # Optionally, display the number of faces detected
    st.markdown(f"#### Number of faces detected: {len(faces)}")
    # st.markdown(f"<h4 style='text-align: center;'>Number of faces detected: {len(faces)}</h1>", unsafe_allow_html=True)
    # st.markdown(f"<h4 style='text-align: center;'>Mask status : {results}</h1>", unsafe_allow_html=True)

    st.markdown(f"#### Mask status : {results}")

