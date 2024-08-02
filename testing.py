import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import threading


def load_model(weights, config, names):
    net = cv2.dnn.readNet(weights, config)
    with open(names, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers


def detect_objects(img, net, output_layers, classes, conf_threshold=0.5, nms_threshold=0.4):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return [boxes[i] for i in indexes], [classes[class_ids[i]] for i in indexes], [confidences[i] for i in indexes]


# Load models
bib_net, bib_classes, bib_output_layers = load_model(
    "weights-classes/RBNR_custom-yolov4-tiny-detector_best.weights",
    "weights-classes/RBNR_custom-yolov4-tiny-detector.cfg",
    "weights-classes/RBRN_obj.names"
)

svhn_net, svhn_classes, svhn_output_layers = load_model(
    "weights-classes/SVHN_custom-yolov4-tiny-detector_best.weights",
    "weights-classes/SVHN_custom-yolov4-tiny-detector.cfg",
    "weights-classes/SVHN_obj.names"
)


def process_frame(frame):
    global bib_classes, svhn_classes  # Make these global

    # Check if frame is valid
    if frame is None or frame.size == 0:
        print("Invalid frame")
        return frame

    # Detect bib
    bib_boxes, bib_classes_detected, bib_confidences = detect_objects(frame, bib_net, bib_output_layers, bib_classes,
                                                                      conf_threshold=0.3)

    # Process each detected bib
    for box in bib_boxes:
        x, y, w, h = box

        # Ensure bib ROI is within frame boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        # Check if bib ROI is valid
        if w <= 0 or h <= 0:
            print("Invalid bib ROI")
            continue

        bib_roi = frame[y:y + h, x:x + w]

        # Check if bib_roi is valid
        if bib_roi is None or bib_roi.size == 0:
            print("Empty bib ROI")
            continue

        try:
            # Detect numbers on the bib
            number_boxes, number_classes, number_confidences = detect_objects(bib_roi, svhn_net, svhn_output_layers,
                                                                              svhn_classes, conf_threshold=0.3)

            # Sort detected numbers based on x-coordinate (left to right)
            sorted_numbers = sorted(zip(number_boxes, number_classes, number_confidences), key=lambda item: item[0][0])

            # Concatenate detected numbers
            bib_number = ''.join([num_class for _, num_class, _ in sorted_numbers])

            # Draw bib bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Display the full bib number
            cv2.putText(frame, f"Bib: {bib_number}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        except cv2.error as e:
            print(f"OpenCV error: {str(e)}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")

    return frame

def resize_frame(frame, max_width, max_height):
    h, w = frame.shape[:2]
    aspect = w / h
    if max_width / aspect <= max_height:
        new_w = max_width
        new_h = int(max_width / aspect)
    else:
        new_w = int(max_height * aspect)
        new_h = max_height
    return cv2.resize(frame, (new_w, new_h))


def process_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
    if file_path:
        img = cv2.imread(file_path)
        processed_img = process_frame(img)
        cv2.namedWindow("Processed Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Processed Image", processed_img)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty("Processed Image", cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()


def process_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if file_path:
        cap = cv2.VideoCapture(file_path)
        cv2.namedWindow("Processed Video", cv2.WINDOW_NORMAL)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = process_frame(frame)

            window_width = cv2.getWindowImageRect("Processed Video")[2]
            window_height = cv2.getWindowImageRect("Processed Video")[3]
            resized_frame = resize_frame(processed_frame, window_width, window_height)

            cv2.imshow("Processed Video", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Processed Video", cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyAllWindows()


def process_live_feed():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)

        window_width = cv2.getWindowImageRect("Live Feed")[2]
        window_height = cv2.getWindowImageRect("Live Feed")[3]
        resized_frame = resize_frame(processed_frame, window_width, window_height)

        cv2.imshow("Live Feed", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Live Feed", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


def run_threaded(func):
    threading.Thread(target=func).start()


# Create the main window
root = tk.Tk()
root.title("Bib Number Detection")
root.geometry("400x300")  # Set a larger initial size

# Create buttons with larger font
button_font = ('Arial', 14)
image_button = tk.Button(root, text="Process Image", command=lambda: run_threaded(process_image), font=button_font)
image_button.pack(pady=20, fill=tk.X, padx=50)

video_button = tk.Button(root, text="Process Video", command=lambda: run_threaded(process_video), font=button_font)
video_button.pack(pady=20, fill=tk.X, padx=50)

live_feed_button = tk.Button(root, text="Process Live Feed", command=lambda: run_threaded(process_live_feed), font=button_font)
live_feed_button.pack(pady=20, fill=tk.X, padx=50)

# Run the Tkinter event loop
root.mainloop()