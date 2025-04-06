import cv2
import numpy as np
import time
from centroidtracker import CentroidTracker

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Init Tracker
tracker = CentroidTracker(max_disappeared=30)
object_timers = {}

# Open webcam
# cap = cv2.VideoCapture("C:/Users/Arunkumar/Downloads/4K Road traffic video for object detection and tracking.mp4")
cap = cv2.VideoCapture("http://192.168.1.33:4747/video")




while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    current_time = time.time()

    # Detect objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

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

                # Filter only on specific objects (optional)
                if classes[class_id] in ["person", "car", "tractor", "motorbike", "bicycle"]:
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    rects = []

    if len(indexes) > 0:
        for i in indexes.flatten():
            rects.append(boxes[i])

    # Update tracker and get object IDs with their centroids
    objects = tracker.update(rects)

    if len(indexes) > 0:
        for i in indexes:
            i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            # your drawing code

        # Draw the bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add label and confidence
        text = f"{label}: {int(confidence * 100)}%"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    for object_id, centroid in objects.items():
        # Draw object ID and timer
        cx, cy = centroid

        # Track entry time
        if object_id not in object_timers:
            object_timers[object_id] = {
                "enter_time": current_time,
                "exit_time": None
            }

        # Update exit time continuously
        object_timers[object_id]["exit_time"] = current_time
        duration = current_time - object_timers[object_id]["enter_time"]

        # Draw circle & info
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"ID {object_id} - {int(duration)}s", (cx - 20, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Check for disappeared objects
    existing_ids = set(objects.keys())
    tracked_ids = set(object_timers.keys())

    # At top of script
    bike_count = 0
    object_labels = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))  # ✅ Resize inside loop

        height, width, _ = frame.shape
        current_time = time.time()

        # [ ... your detection code remains the same ... ]

        # Draw and track
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = f"{label}: {int(confidence * 100)}%"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for object_id, centroid in objects.items():
            cx, cy = centroid

            if object_id not in object_timers:
                object_timers[object_id] = {
                    "enter_time": current_time,
                    "exit_time": None
                }

            object_timers[object_id]["exit_time"] = current_time
            duration = current_time - object_timers[object_id]["enter_time"]

            # Store label if not already set
            if object_id not in object_labels:
                object_labels[object_id] = label

            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"ID {object_id} - {int(duration)}s", (cx - 20, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Track removed objects
        existing_ids = set(objects.keys())
        tracked_ids = set(object_timers.keys())

        for old_id in list(tracked_ids - existing_ids):
            enter = object_timers[old_id]["enter_time"]
            exit = object_timers[old_id]["exit_time"]
            label = object_labels.get(old_id, "")
            if exit and enter:
                print(f"[INFO] Object ID {old_id} ({label}) stayed for {exit - enter:.2f} seconds")
                if label in ["motorbike", "bicycle"]:
                    bike_count += 1
                    print(f"[BIKE COUNT] Total passed bikes: {bike_count}")
            del object_timers[old_id]
            object_labels.pop(old_id, None)

        # Draw count
        cv2.putText(frame, f"Bikes Passed: {bike_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Object Tracking", frame)

        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()
