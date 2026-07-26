import cv2
import mediapipe as mp
import requests
import time

ESP32_IP = "http://10.121.137.218"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# ✅ STATE MEMORY
prev_thumb = None
prev_index = None
prev_middle = None
prev_ring = None
prev_pinky = None

def control_led(endpoint):
    url = f"{ESP32_IP}/led/{endpoint}"
    try:
        requests.get(url, timeout=0.5)
        print(f"Sent: {endpoint}")
    except Exception as e:
        print(f"❌ LED Error: {e}")

def count_fingers(hand_landmarks):
    global prev_thumb, prev_index, prev_middle, prev_ring, prev_pinky

    thumb_up = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
    index_up = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    middle_up = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring_up = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
    pinky_up = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y

    # ✅ Only send when state changes

    if thumb_up != prev_thumb:
        control_led("thumb/on" if thumb_up else "thumb/off")
        prev_thumb = thumb_up

    if index_up != prev_index:
        control_led("index/on" if index_up else "index/off")
        prev_index = index_up

    if middle_up != prev_middle:
        control_led("middle/on" if middle_up else "middle/off")
        prev_middle = middle_up

    if ring_up != prev_ring:
        control_led("ring/on" if ring_up else "ring/off")
        prev_ring = ring_up

    if pinky_up != prev_pinky:
        control_led("pinky/on" if pinky_up else "pinky/off")
        prev_pinky = pinky_up


# CAMERA
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not opening")
    exit()
else:
    print("✅ Camera started")

print("🔄 Starting main loop...")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            count_fingers(hand_landmarks)

    cv2.imshow("Hand Gesture Recognition", frame)

    time.sleep(0.05)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
