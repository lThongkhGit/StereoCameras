import glob
import cv2 as cv
import numpy as np
import Calculation3D as Cal3D
import HandRecognition
import pandas as pd

path_glob = 'Data/Clean/Test_Gauche_27-06/Data/8_-1_3_NA/Result/'
path_vid_1 = path_glob + 'Camera1.mp4'
path_vid_2 = path_glob + 'Camera2.mp4'
show_vid = True
path_calib = 'Data/Calib'

frame1_ref = cv.imread(path_calib + '/Synced/1_ref.jpg')
frame2_ref = cv.imread(path_calib + '/Synced/2_ref.jpg')

# Get each camera extrinsic parameters
mtx1, dist1 = Cal3D.calibrate_camera(images_folder = path_calib + '/Cam1/*')
mtx2, dist2 = Cal3D.calibrate_camera(images_folder = path_calib + '/Cam2/*')

# Get synched shots from the two cameras for stereo calibration
images_names = glob.glob(path_calib + '/Synced/*')
images_names = sorted(images_names)
c1_images_names = images_names[:len(images_names) // 2]
c2_images_names = images_names[len(images_names) // 2:]
c1_images = []
c2_images = []
for im1, im2 in zip(c1_images_names, c2_images_names):
    _im = cv.imread(im1, 1)
    c1_images.append(_im)

    _im = cv.imread(im2, 1)
    c2_images.append(_im)

# Stereo Calibration
R, T = Cal3D.stereo_calibrate(c1_images, c2_images, mtx1, dist1, mtx2, dist2)
#print("R : " + str(R))
#print("T : " + str(T))
P1, P2 = Cal3D.triangulate(mtx1, mtx2, R, T)

# Change coordinate system
u, v, origin = Cal3D.get_new_coordinate_system_base(frame1_ref, frame2_ref, P1, P2)


def main_wrist():
    points_3D = Cal3D.get_wrist_3D_coordinates(path_vid_1, path_vid_2, P1, P2, show_vid)
    #print("Points_3D size : " + str(len(points_3D)))
    #print(points_3D)
    points_3D_new_ref = []
    for i in range(len(points_3D)):
        points_3D_new_ref.append(Cal3D.change_coordinates(u, v, origin, points_3D[i]))

    #print("points_3D_new_ref size : " + str(len(points_3D_new_ref)))
    #print(points_3D_new_ref)

    smooth_points_3D = Cal3D.smooth_curve(points_3D_new_ref)

    #Cal3D.plot_3D_points_plotly(points_3D_new_ref)

    Cal3D.plot_3D_points_plotly(points_3D_new_ref, '')
    Cal3D.plot_3D_points_plotly(smooth_points_3D, path_glob + 'Wrist_plot.html')


def main_full_hand():
    landmarks_coordinates_3D = Cal3D.get_full_hand_3D_coordinates(path_vid_1, path_vid_2, P1, P2, show_vid)
    #print(landmarks_coordinates_3D[0])
    landmarks_coordinates_3D_new_ref = []
    for i in range(len(HandRecognition.HAND_LANDMARK)):
        landmarks_coordinates_3D_new_ref.append(Cal3D.change_coordinates_list(u, v, origin, landmarks_coordinates_3D[i]))

    landmarks_coordinates_3D_new_ref = np.array(Cal3D.fill_empty_data(landmarks_coordinates_3D_new_ref))
    #print(landmarks_coordinates_3D_new_ref[0])

    x = []
    y = []
    z = []
    for i in range(len(landmarks_coordinates_3D_new_ref[0])):
        x.append(landmarks_coordinates_3D_new_ref[0][i][0])
        y.append(landmarks_coordinates_3D_new_ref[0][i][1])
        z.append(landmarks_coordinates_3D_new_ref[0][i][2])
    df_wrist = pd.DataFrame()
    df_wrist['x'] = x
    df_wrist['y'] = y
    df_wrist['z'] = z
    print(df_wrist.head())
    #df_wrist = df_wrist.rename(columns={"0": "x", "1": "y", "2": "z"})
    df_wrist.to_csv(path_glob + 'Wrist.csv')
    Cal3D.plot_3D_points_plotly(landmarks_coordinates_3D_new_ref[0], path_glob + 'Wrist_plot.html')
    Cal3D.plot_3D_full_hand(landmarks_coordinates_3D_new_ref, path_glob + 'hand.gif')

main_full_hand()