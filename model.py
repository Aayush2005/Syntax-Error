from ultralytics import YOLO
import cv2
import numpy as np
from utils.centroidtracker import CentroidTracker
from utils.object_trackable import TrackableObject
import argparse
from deepface import DeepFace
from datetime import datetime
import json

# Initializing Parameters
confThreshold = 0.7
inpWidth, inpHeight = 640, 640

# Custom function to handle video input
def video_input(value):
    try:
        return int(value)
    except ValueError:
        return value

# Argument Parser Setup
parser = argparse.ArgumentParser(description="Women safety")
parser.add_argument('--video', type=video_input, default=0, help='Video file path, URL, or webcam index')
args = parser.parse_args()

# Load YOLOv8 Model
print("Loading model...")
model = YOLO('yolov8n.pt')

# Centroid Tracker Initialization
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackableObjects = {}
totalDown = 0
totalUp = 0

# Video Input Setup
cap = cv2.VideoCapture(args.video)  # Use input from argparse
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the input video frame rate
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: 
    fps = 30
print(f"Input video FPS: {fps}")

# Function to Compute IoU
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

# Function to Merge Overlapping Boxes
def merge_boxes(boxes, iou_threshold=0.5):
    merged_boxes = []
    while boxes:
        box = boxes.pop(0)
        to_merge = []
        for other_box in boxes:
            if compute_iou(box, other_box) > iou_threshold:
                to_merge.append(other_box)
        for merge_box in to_merge:
            boxes.remove(merge_box)
            box[0] = min(box[0], merge_box[0])
            box[1] = min(box[1], merge_box[1])
            box[2] = max(box[2], merge_box[2])
            box[3] = max(box[3], merge_box[3])
        merged_boxes.append(box)
    return merged_boxes

# Gender Prediction using DeepFace
def predict_gender(cropped_person):
    try:
        # Convert OpenCV image (BGR) to RGB
        rgb_person = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
        
        # Perform gender analysis
        results = DeepFace.analyze(img_path=rgb_person, actions=['gender'], enforce_detection=False)
        
        # Ensure results are a dictionary
        if isinstance(results, list):
            results = results[0] 
        
        # Get dominant gender
        gender = max(results['gender'], key=results['gender'].get)
        print(f"Predicted Gender: {gender}")
        
        # Return the gender
        return gender

    except Exception as e:
        print(f"Error during gender prediction: {e}")
        return "Unknown"

# Counting Function
def counting(objects, frame):
    global totalDown, totalUp
    frameHeight = frame.shape[0]

    for (objectID, centroid) in objects.items():
        to = trackableObjects.get(objectID, None)

        if to is None:
            to = TrackableObject(objectID, centroid)
        else:
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            if not to.counted:
                if direction < 0 and frameHeight // 2 - 30 <= centroid[1] <= frameHeight // 2 + 30:
                    totalUp += 1
                    to.counted = True
                elif direction > 0 and frameHeight // 2 - 30 <= centroid[1] <= frameHeight // 2 + 30:
                    totalDown += 1
                    to.counted = True

        trackableObjects[objectID] = to

# Initialize JSON data dictionary
json_data = {
    "people_count": 0,
    "men": 0,
    "women": 0,
    "lone_women": 0
}

# Function to check if it is night time (6 PM to 6 AM)
def is_night_time():
    current_hour = datetime.now().hour
    return current_hour >= 18 or current_hour < 6

# Function to check if only one woman is detected (lone woman)
def is_lone_person(objects):
    return len(objects) == 1

# Processing Video Frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Done processing!")
        break

    # YOLOv8 Inference
    results = model(frame)
    detections = results[0].boxes
    rects = []

    for box in detections:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf[0]
        class_id = int(box.cls[0])

        # Only process "person" class (class_id == 0)
        if class_id == 0 and conf > confThreshold:
            rects.append([x1, y1, x2, y2])

    # Merge overlapping bounding boxes
    rects = merge_boxes(rects)

    # Initialize counters for gender
    male_count = 0
    female_count = 0

    # Process each detected person
    for i, rect in enumerate(rects):
        x1, y1, x2, y2 = rect
        cropped_person = frame[y1:y2, x1:x2]  # Crop the person from the frame

        # Predict gender
        gender = predict_gender(cropped_person)

        # Count genders
        if gender == 'Man':
            male_count += 1
        elif gender == 'Woman':
            female_count += 1

    # Update Centroid Tracker and Count
    objects = ct.update([(x1, y1, x2, y2) for x1, y1, x2, y2 in rects])
    counting(objects, frame)

    # Check for lone woman at night
    lone_women_count = 0
    if is_night_time() and is_lone_person(objects) and female_count == 1:
        print("Lone woman detected at night!")
        lone_women_count += 1
        # Save the frame of lone woman detected at night
        cv2.imwrite(f"lone_woman_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", frame)

    # Display Number of People and Gender Counts
    cv2.putText(frame, f"People in Frame: {len(objects)}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Males: {male_count} | Females: {female_count}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Update JSON data
    json_data['people_count'] = len(objects)
    json_data['men'] = male_count
    json_data['women'] = female_count
    json_data['lone_women'] = lone_women_count

    # Save JSON data periodically if needed
    with open('people_count.json', 'w') as json_file:
        json.dump(json_data, json_file)

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
