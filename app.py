import streamlit as st
import cv2
import numpy as np
import os

# Load YOLO
@st.cache_resource
def load_yolo():
    # Assuming the config, weights, and names files are in the same directory as the script
    base_path = os.path.dirname(__file__)
    config_path = os.path.join(base_path, "yolov3.cfg")
    weights_path = os.path.join(base_path, "yolov3.weights")
    names_path = os.path.join(base_path, "coco.names")

    try:
        # Check if files exist
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        if not os.path.exists(names_path):
            raise FileNotFoundError(f"Names file not found: {names_path}")

        net = cv2.dnn.readNet(weights_path, config_path)
        with open(names_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        
        layer_names = net.getLayerNames()
        unconnected_out_layers = net.getUnconnectedOutLayers()

        # Debugging information
        #st.write("Layer names:", layer_names)
        #st.write("Unconnected out layers:", unconnected_out_layers)

        if len(unconnected_out_layers) == 0:
            st.error("No unconnected output layers found. Check your YOLO model files.")
            return None, None, None, None

        output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        return net, classes, output_layers, colors
    except Exception as e:
        st.error(f"Error loading YOLO: {str(e)}")
        return None, None, None, None

def detect_objects(img, net, output_layers):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    return outs

def draw_labels(outs, img, classes, colors):
    height, width, channels = img.shape
    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), font, 1, color, 2)
    return img

def main():
    st.title("Real-time Object Detection")
    st.write("Upload an image or use your webcam for real-time object detection using YOLO.")

    net, classes, output_layers, colors = load_yolo()

    if net is None:
        return

    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose Input Source", ("Image", "Webcam"))

    if option == "Image":
        st.sidebar.write("Upload an Image")
        uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write("")
            st.write("Detecting objects...")
            outs = detect_objects(image, net, output_layers)
            image = draw_labels(outs, image, classes, colors)
            st.image(image, caption='Processed Image', use_column_width=True)

    elif option == "Webcam":
        run = st.checkbox('Run')
        FRAME_WINDOW = st.image([])

        cap = cv2.VideoCapture(0)

        while run:
            _, frame = cap.read()
            outs = detect_objects(frame, net, output_layers)
            frame = draw_labels(outs, frame, classes, colors)
            FRAME_WINDOW.image(frame)
        else:
            cap.release()

if __name__ == "__main__":
    main()
    
# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<footer><p>Developed by Akhil Sudhakaran</p></footer>", unsafe_allow_html=True)
