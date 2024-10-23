import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

# Load YOLOv11 model
@st.cache_resource
def load_yolo_v11():
    try:
        model = YOLO("yolov8n.pt")  # YOLOv11 is accessed via ultralytics package
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv11: {str(e)}")
        return None

def detect_objects_v11(img, model):
    results = model.predict(img)
    return results

def draw_labels_v11(results, img, class_names):
    annotated_frame = img.copy()
    boxes = results[0].boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Ensure coordinates are integers
        label_id = int(box.cls)  # Convert label to integer
        conf = float(box.conf)  # Convert confidence to float
        label = class_names[label_id] if label_id < len(class_names) else "Unknown"

        # Draw the bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Calculate text size for background rectangle
        text = f'{label} {conf:.2f}'
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

        # Set text position inside the box
        text_offset_x, text_offset_y = x1 + 5, y1 + text_height + 5
        
        # Ensure the text background rectangle is within the bounding box
        box_coords = (
            (x1, y1),
            (x1 + text_width + 10, y1 + text_height + 10)
        )

        # Draw the background rectangle
        cv2.rectangle(annotated_frame, box_coords[0], box_coords[1], (0, 255, 0), cv2.FILLED)

        # Draw the text
        cv2.putText(annotated_frame, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    return annotated_frame

def main():
    st.title("Real-time Object Detection")
    st.write("Upload an image, a video file, or use your webcam for real-time object detection using YOLOv11.")
    model = load_yolo_v11()
    if model is None:
        return

    # Get class names from the model
    class_names = model.names if hasattr(model, 'names') else []

    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose Input Source", ("Image", "Video", "Webcam"))

    if option == "Image":
        st.sidebar.write("Upload an Image")
        uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption='Uploaded Image', use_column_width=True)
            st.write("")
            st.write("Detecting objects...")
            results = detect_objects_v11(image, model)
            annotated_image = draw_labels_v11(results, image, class_names)
            # Convert BGR to RGB before displaying
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            st.image(annotated_image_rgb, caption='Processed Image', use_column_width=True)

    elif option == "Video":
        st.sidebar.write("Upload a Video")
        uploaded_video = st.sidebar.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())

            cap = cv2.VideoCapture(tfile.name)
            FRAME_WINDOW = st.image([])
            run = st.checkbox('Run')
            while run and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read frame from video file.")
                    break

                results = detect_objects_v11(frame, model)
                annotated_frame = draw_labels_v11(results, frame, class_names)
                # Convert BGR to RGB before displaying
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(annotated_frame_rgb)

            cap.release()

    elif option == "Webcam":
        run = st.checkbox('Run')
        FRAME_WINDOW = st.image([])
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open webcam.")
                return
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read frame from webcam.")
                    break

                results = detect_objects_v11(frame, model)
                annotated_frame = draw_labels_v11(results, frame, class_names)
                # Convert BGR to RGB before displaying
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(annotated_frame_rgb)

            cap.release()
        except Exception as e:
            st.error(f"Error accessing webcam: {str(e)}")
            return

if __name__ == "__main__":
    main()

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<footer><p>Developed by Akhil Sudhakaran</p></footer>", unsafe_allow_html=True)
