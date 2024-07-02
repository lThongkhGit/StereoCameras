import os.path
import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

'''
This script deals with the hand recognition. Extracts the coordinate of the landmarks of the hand from
a video. Uses the IA from OpenCV mediapipe to detect the hand. One should visit the mediapipe documentation
to understand the naming and connections of the landmarks.
'''

# Initialize mediapipe hands module
mphands = mp.solutions.hands
mpdrawing = mp.solutions.drawing_utils

# Set the desired window width and height
winwidth = 1920
winheight = 1080
HAND_LANDMARK = [mphands.HandLandmark.WRIST, mphands.HandLandmark.THUMB_CMC, mphands.HandLandmark.THUMB_MCP,
                 mphands.HandLandmark.THUMB_IP, mphands.HandLandmark.THUMB_TIP,
                 mphands.HandLandmark.INDEX_FINGER_MCP, mphands.HandLandmark.INDEX_FINGER_PIP,
                 mphands.HandLandmark.INDEX_FINGER_DIP, mphands.HandLandmark.INDEX_FINGER_TIP,
                 mphands.HandLandmark.MIDDLE_FINGER_MCP, mphands.HandLandmark.MIDDLE_FINGER_PIP,
                 mphands.HandLandmark.MIDDLE_FINGER_DIP, mphands.HandLandmark.MIDDLE_FINGER_TIP,
                 mphands.HandLandmark.RING_FINGER_MCP, mphands.HandLandmark.RING_FINGER_PIP,
                 mphands.HandLandmark.RING_FINGER_DIP, mphands.HandLandmark.RING_FINGER_TIP,
                 mphands.HandLandmark.PINKY_MCP, mphands.HandLandmark.PINKY_PIP, mphands.HandLandmark.PINKY_DIP,
                 mphands.HandLandmark.PINKY_TIP]

CONNECTIONS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
               [9, 13], [13, 14],
               [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [17, 0]]
CONNECTIONS_COLOR = ['orange', 'orange', 'orange', 'orange', 'red', 'yellow', 'yellow', 'yellow', 'red', 'lime', 'lime',
                     'lime',
                     'red', 'cyan', 'cyan', 'cyan', 'red', 'blue', 'blue', 'blue', 'red']

vidpath = 'Data/Clean/Test_gauche_27-06/Data/8_-1_3_NA/Result/Camera1.mp4'


# Returns the list of coordinates of the wrist from a given video
def read_video_coordinate(path_cam, show_vid):
    if os.path.exists(path_cam):
        # Initialize video capture and writer
        vidcap = cv2.VideoCapture(path_cam)
        output = cv2.VideoWriter(path_cam[:-4] + '_detected.mp4', cv2.VideoWriter_fourcc(*'MPEG'), 30, (1920, 1080))

        # Initialize hand tracking
        with mphands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

            x = []
            y = []
            while vidcap.isOpened():
                ret, frame = vidcap.read()
                if not ret:
                    break

                # Convert the BGR image to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame for hand tracking
                processFrames = hands.process(rgb_frame)

                if processFrames.multi_hand_landmarks:
                    for lm in processFrames.multi_hand_landmarks:
                        # Draw landmarks on the frame
                        mpdrawing.draw_landmarks(frame, lm, mphands.HAND_CONNECTIONS)
                        x.append(lm.landmark[mphands.HandLandmark.WRIST].x * 1920)
                        y.append(lm.landmark[mphands.HandLandmark.WRIST].y * 1080)
                else:
                    x.append(None)
                    y.append(None)
                    print("No hand detected!")

                # Resize the frame to the desired window size
                resized_frame = cv2.resize(frame, (winwidth, winheight))
                output.write(resized_frame)
                # Display the resized frame
                if show_vid:
                    cv2.imshow('Hand Tracking', resized_frame)

                # Exit loop by pressing 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Release the video capture and close windows
        vidcap.release()
        output.release()
        cv2.destroyAllWindows()
        return [x, y]
    else:
        print("Video path wrong")
        return


# Returns a list of lists of coordinates of all the landmarks of the hand from a given video
def read_video_coordinate_full_hand(path_cam, show_vid):
    if os.path.exists(path_cam):
        # Initialize video capture
        vidcap = cv2.VideoCapture(path_cam)
        output = cv2.VideoWriter(path_cam[:-4] + '_detected.mp4', cv2.VideoWriter_fourcc(*'MPEG'), 30, (1920, 1080))

        landmarks_coordinates = []
        for i in range(len(HAND_LANDMARK)):
            landmarks_coordinates.append([])

        # Initialize hand tracking
        with mphands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

            while vidcap.isOpened():
                ret, frame = vidcap.read()
                if not ret:
                    break

                # Convert the BGR image to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame for hand tracking
                processFrames = hands.process(rgb_frame)

                if processFrames.multi_hand_landmarks:
                    for lm in processFrames.multi_hand_landmarks:
                        # Draw landmarks on the frame
                        mpdrawing.draw_landmarks(frame, lm, mphands.HAND_CONNECTIONS)
                        append_landmarks_coordinates(lm, landmarks_coordinates)
                else:
                    append_landmarks_coordinates_None(landmarks_coordinates)
                    print("No hand detected!")

                # Resize the frame to the desired window size
                resized_frame = cv2.resize(frame, (winwidth, winheight))
                output.write(resized_frame)
                # Display the resized frame
                if show_vid:
                    cv2.imshow('Hand Tracking', resized_frame)

                # Exit loop by pressing 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Release the video capture and close windows
        vidcap.release()
        output.release()
        cv2.destroyAllWindows()
        return landmarks_coordinates
    else:
        print("Video path wrong")
        return


def append_landmarks_coordinates(lm, landmark_coordinates):
    for i in range(len(HAND_LANDMARK)):
        landmark_coordinates[i].insert(len(landmark_coordinates[i]),
                                       [lm.landmark[HAND_LANDMARK[i]].x * winwidth,
                                         lm.landmark[HAND_LANDMARK[i]].y * winheight])


def append_landmarks_coordinates_None(landmmark_coordinates):
    for i in range(len(HAND_LANDMARK)):
        landmmark_coordinates[i].insert(len(landmmark_coordinates[i]),
                                        [None, None])


'''landmarks = read_video_coordinate_full_hand(vidpath, False)
print(len(landmarks))
print(len(landmarks[0]))
print(landmarks[0])
print(len(landmarks[0][0]))
print(landmarks[0][0])'''

'''points = read_video_coordinate(vidpath, True)
points = np.array(points)
print(points.shape)
print(points)'''
