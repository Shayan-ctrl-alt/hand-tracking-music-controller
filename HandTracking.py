import cv2
import mediapipe as mp
import pygame
import time

# Initialize mediapipe and pygame
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Initialize the mixer for playing music
pygame.mixer.init()
playlist = ["Your Song File", "Your Song File", "Your Song File"]
current_song_index = 0
pygame.mixer.music.load(playlist[current_song_index])
pygame.mixer.music.play(-1)

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

previous_x_position = None
swipe_time = 0

def play_current_song():
    pygame.mixer.music.load(playlist[current_song_index])
    pygame.mixer.music.play(-1)

def detect_swipe(current_x,previous_x):
    global swipe_time
    if current_x > previous_x + 0.2 and time.time() - swipe_time > 1:
        return "next"
    elif current_x < previous_x - 0.2 and time.time() - swipe_time > 1:
        return "previous"
    return None


# Function to calculate the volume based on distance between thumb and index finger
def calculate_volume(thumb_index_distance):
    # Map the distance to a volume range (0.0 to 1.0)
    min_distance = 0.04  # Adjust the minimum distance for volume control
    max_distance = 0.15  # Maximum distance (change this value as needed)

    # Clamp the distance to be between the minimum and maximum
    clamped_distance = max(min(thumb_index_distance, max_distance), min_distance)

    # Normalize the distance to a value between 0 and 1, and reverse it (closer distance = higher volume)
    normalized_distance = (clamped_distance - min_distance) / (max_distance - min_distance)
    volume = 1.0 - normalized_distance  # Inverted, so smaller distance means higher volume
    return volume


while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
            index_tip_s = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Calculate the distance between thumb and index finger
            thumb_index_distance = ((thumb_tip.x - index_tip_s.x) ** 2 + (thumb_tip.y - index_tip_s.y) ** 2) ** 0.5

            if previous_x_position is not None:
                swipe_directon = detect_swipe(index_tip, previous_x_position)
                if swipe_directon == "next":
                    current_song_index = (current_song_index + 1)% len(playlist)
                    play_current_song()
                    swipe_time = time.time()
                elif swipe_directon == "previous":
                    current_song_index = (current_song_index - 1)%len(playlist)
                    play_current_song()
                    swipe_time = time.time()

            previous_x_position = index_tip
            # Control music play/pause if the fingers are very close
            if thumb_index_distance < 0.02:
                pygame.mixer.music.pause()
            else:
                pygame.mixer.music.unpause()

            # Control the volume if the fingers are at a moderate distance
            if 0.02 < thumb_index_distance < 0.15:  # Distances for volume control range
                volume = calculate_volume(thumb_index_distance)
                pygame.mixer.music.set_volume(volume)

    # Show the video feed with landmarks
    cv2.imshow('Handtracker', frame)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()