import cv2 as cv
import glob
import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt, animation
from scipy.signal import savgol_filter
import HandRecognition as HR

'''
In this script, the first camera should refer to the one on the
right side of the table from the experimenter side.

This script makes all the 3D calculations. Asserts the cameras intrinsec and extrinsic
parameters, triangulates points to get the z coordinate, moves them to a system
coordinate in tone with the table experiment and give visualizers of the extracted data.
'''

CHESS_ROWS = 8
CHESS_COLUMNS = 6
CASE_WIDTH = 2.3

SMOOTH_FILTER_WINDOW = 10
SMOOTH_FILTER_SMOOTHNESS = 3

CHESS_PLAQUE_ROWS = 8
CHESS_PLAQUE_COLUMNS = 6


# Function to get the extrinsic parameters of a camera
# Extrinsic parameters are then used to calibrate the two cameras together.
def calibrate_camera(images_folder):
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        print(imname)
        im = cv.imread(imname, 1)
        images.append(im)

    # criteria used by checkerboard pattern detector.
    # Change this if the code can't find the checkerboard
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    rows = CHESS_ROWS  # number of checkerboard rows.
    columns = CHESS_COLUMNS  # number of checkerboard columns.
    world_scaling = CASE_WIDTH  # change this to the real world square size. Or not.

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints = []  # 2d points in image plane.

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:
            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            # opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows, columns), corners, ret)
            # cv.imshow('img', frame)
            # k = cv.waitKey(100)

            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print("No Chessboard detected")

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', mtx)
    print('distortion coeffs:', dist)
    print('Rs:\n', rvecs)
    print('Ts:\n', tvecs)

    return mtx, dist


# Function to calibrate the two cameras together
# using the parameters output of the single calibration function
# Outputs the Rotation and Translation matrices to put positions
# from the second camera into the coordinate system of the first one.
def stereo_calibrate(cam1_images, cam2_images, mtx1, dist1, mtx2, dist2):
    # criteria used by checkerboard pattern detector.
    # Change this if the code can't find the checkerboard
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    rows = CHESS_ROWS  # number of checkerboard rows.
    columns = CHESS_COLUMNS  # number of checkerboard columns.
    world_scaling = CASE_WIDTH  # change this to the real world square size. Or not.

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = cam1_images[0].shape[1]
    height = cam1_images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints_1 = []  # 2d points in image plane.
    imgpoints_2 = []

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    for frame1, frame2 in zip(cam1_images, cam2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            cv.drawChessboardCorners(frame1, (rows, columns), corners1, c_ret1)
            # cv.imshow('img', frame1)

            cv.drawChessboardCorners(frame2, (rows, columns), corners2, c_ret2)
            # cv.imshow('img2', frame2)
            # k = cv.waitKey(100)

            objpoints.append(objp)
            imgpoints_1.append(corners1)
            # imgpoints_2.append(corners2)
            imgpoints_2.append(corners2[::-1])  # Should not be reversed, but the current camera set up makes it the chessboard cases are read by the second camera in a different order than the first camera
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_1, imgpoints_2, mtx1,
                                                                 dist1, mtx2, dist2, (width, height),
                                                                 criteria=criteria, flags=stereocalibration_flags)
    print("RMSE : " + str(ret))
    return R, T


def triangulate(mtx1, mtx2, R, T):
    # RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    P1 = mtx1 @ RT1  # projection matrix for C1

    # RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis=-1)
    P2 = mtx2 @ RT2  # projection matrix for C2

    return P1, P2


# Calculates the 3D points coordinate in the coordinate system
# of the first camera
def DLT(P1, P2, point1, point2):
    if (not points_null(point1, point2)):
        A = [point1[1] * P1[2, :] - P1[1, :],
             P1[0, :] - point1[0] * P1[2, :],
             point2[1] * P2[2, :] - P2[1, :],
             P2[0, :] - point2[0] * P2[2, :]
             ]
        A = np.array(A).reshape((4, 4))

        B = A.transpose() @ A
        from scipy import linalg
        U, s, Vh = linalg.svd(B, full_matrices=False)

        print('Triangulated point: ' + str(Vh[3, 0:3] / Vh[3, 3]))
        return Vh[3, 0:3] / Vh[3, 3]
    else:
        return [None]


def DLT_list(P1, P2, list1, list2):
    points_3D = []
    for i in range(len(list1)):
        point_3D = DLT(P1, P2, list1[i], list2[i])
        '''if point_3D[0] is not None:
            points_3D.append(point_3D)'''
        points_3D.append(point_3D)
    return points_3D


# We use the shots with the chess plaque as reference
# for the coordinate system. X-Y is the plane of the table (X left to right, Y experimenter side to patient side)
# Z axis the vertical (going upward)
# Outputs the two vectors that serve as basis for the new coordinate system
# and its origin (bottom left corner of the chess)
def get_new_coordinate_system_base(camera1_shot, camera2_shot, P1, P2):
    gray1 = cv.cvtColor(camera1_shot, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(camera2_shot, cv.COLOR_BGR2GRAY)
    c_ret1, corners1 = cv.findChessboardCorners(gray1, (CHESS_PLAQUE_ROWS, CHESS_PLAQUE_COLUMNS), None)
    c_ret2, corners2 = cv.findChessboardCorners(gray2, (CHESS_PLAQUE_ROWS, CHESS_PLAQUE_COLUMNS), None)

    corner_1_bottom_left = corners1[0][0]
    corner_1_top_left = corners1[CHESS_PLAQUE_ROWS - 1][0]
    corner_1_bottom_right = corners1[-CHESS_PLAQUE_ROWS][0]
    corner_2_bottom_left = corners2[-1][0]  # With different angles of cameras, Chessboard detections makes up different order of cases
    corner_2_top_left = corners2[-CHESS_PLAQUE_ROWS][0]
    corner_2_bottom_right = corners2[CHESS_PLAQUE_ROWS - 1][0]
    '''print("corner_1_bottom_left " + str(corner_1_bottom_left))
    print("corner_1_top_left " + str(corner_1_top_left))
    print("corner_1_bottom_right " + str(corner_1_bottom_right))
    print("corner_2_bottom_left " + str(corner_2_bottom_left))
    print("corner_2_top_left " + str(corner_2_top_left))
    print("corner_2_bottom_right " + str(corner_2_bottom_right))'''

    corner_1_bottom_left_3d = DLT(P1, P2, corner_1_bottom_left, corner_2_bottom_left)
    corner_1_top_left_3d = DLT(P1, P2, corner_1_top_left, corner_2_top_left)
    corner_1_bottom_right_3d = DLT(P1, P2, corner_1_bottom_right, corner_2_bottom_right)

    '''print("corner_1_bottom_left_3d " + str(corner_1_bottom_left_3d))
    print("corner_1_top_left_3d " + str(corner_1_top_left_3d))
    print("corner_1_bottom_right_3d " + str(corner_1_bottom_right_3d))
    cv.drawChessboardCorners(frame1, (CHESS_PLAQUE_ROWS, CHESS_PLAQUE_COLUMNS), corners1, c_ret1)
    cv.imshow('img', frame1)'''

    u = np.subtract(corner_1_bottom_right_3d, corner_1_bottom_left_3d)
    v = np.subtract(corner_1_top_left_3d, corner_1_bottom_left_3d)
    print("u " + str(u) + ", v " + str(v))
    return u, v, corner_1_bottom_left_3d


def change_coordinates(u, v, origin, point):
    # Normalization
    u_normalized = u / np.linalg.norm(u)
    v_normalized = v / np.linalg.norm(v)
    w_normalized = np.cross(u_normalized, v_normalized)

    # Project on the new coordinate system
    if point[0] is not None:
        new_x = np.dot(u_normalized, np.subtract(point, origin))
        new_y = np.dot(v_normalized, np.subtract(point, origin))
        new_z = np.dot(w_normalized, np.subtract(point, origin))
    else:
        return [None, None, None]
    print("Changed coordinates from " + str(point) + " to " + str([new_x, new_y, new_z]))
    return [new_x, new_y, new_z]


def change_coordinates_list(u, v, origin, points_list):
    result = []
    for i in range(len(points_list)):
        result.append(change_coordinates(u, v, origin, points_list[i]))
    return result


def get_wrist_3D_coordinates(path_cam_1, path_cam_2, P1, P2, show_vid):
    coordinates_1 = HR.read_video_coordinate(path_cam_1, show_vid)
    coordinates_2 = HR.read_video_coordinate(path_cam_2, show_vid)

    print("Coordinates 1 x and y sizes : " + str(len(coordinates_1[0])) + ", " + str(len(coordinates_1[1])))
    print("Coordinates 2 x and y sizes : " + str(len(coordinates_2[0])) + ", " + str(len(coordinates_2[1])))

    # Convert coordinates to 3D using the coordinates from the two cameras and their parameter matrices
    points_3D = []
    length = min(len(coordinates_1[0]), len(coordinates_2[0]))
    for i in range(length):
        _p3d = DLT(P1, P2, [coordinates_1[0][i], coordinates_1[1][i]], [coordinates_2[0][i], coordinates_2[1][i]])
        if _p3d[0] is not None:
            points_3D.append(_p3d)
    points_3D = np.array(points_3D)
    # print(points_3D)

    return points_3D


def get_full_hand_3D_coordinates(path_cam_1, path_cam_2, P1, P2, show_vid):
    coordinates_1 = HR.read_video_coordinate_full_hand(path_cam_1, show_vid)
    coordinates_2 = HR.read_video_coordinate_full_hand(path_cam_2, show_vid)

    landmarks_coordinates_3D = []
    for i in range(len(HR.HAND_LANDMARK)):
        landmarks_coordinates_3D.append(DLT_list(P1, P2, coordinates_1[i], coordinates_2[i]))

    return landmarks_coordinates_3D


def plot_3D_points_plotly(points_3D, save_path):
    # Prepare arrays x, y, z
    x = []
    y = []
    z = []
    for i in range(len(points_3D)):
        x.append(points_3D[i][0])
        y.append(points_3D[i][1])
        z.append(points_3D[i][2])

    if None in z:
        color = [0] * len(z)
    else:
        color = z
    fig = go.Figure(data=go.Scatter3d(x=x, y=y, z=z,
                                      marker=dict(size=4, color=color, colorscale='Viridis'),
                                      line=dict(color='darkblue', width=2)))

    fig.update_layout(width=800, height=700, autosize=False,
                      scene=dict(
                          camera=dict(
                              up=dict(x=0, y=0, z=1),
                              eye=dict(x=0, y=-1.0707, z=0.7)),
                          aspectratio=dict(x=1, y=1, z=0.7),
                          aspectmode='manual'), )
    if save_path != '':
        fig.write_html(save_path)
    fig.show()


def plot_3D_full_hand(landmarks_coordinates_3D, save_path):
    x = []
    y = []
    z = []
    print("Is plotting full hand")

    for i in range(len(landmarks_coordinates_3D[0])):
        for j in range(len(landmarks_coordinates_3D)):
            x.append(landmarks_coordinates_3D[j][i][0])
            y.append(landmarks_coordinates_3D[j][i][1])
            z.append(landmarks_coordinates_3D[j][i][2])

    t = np.array([np.ones(len(landmarks_coordinates_3D)) * i for i in range(len(landmarks_coordinates_3D[0]))]).flatten()
    df = pd.DataFrame({"time": t, "x": x, "y": y, "z": z})

    def update_graph(num):
        data = df[df['time'] == num]
        graph._offsets3d = (data.x, data.y, data.z)
        for i in range(len(lines)):
            lines[i].set_data_3d(np.array([landmarks_coordinates_3D[HR.CONNECTIONS[i][0]][num], landmarks_coordinates_3D[HR.CONNECTIONS[i][1]][num]]).T)
        title.set_text('3D Test, time={}'.format(num))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('3D Test')

    lines = [ax.plot([], [], [], color=HR.CONNECTIONS_COLOR[i])[0] for i in range(len(HR.CONNECTIONS_COLOR))]

    # -20 15 Main Droite  / Main gauche 0 35
    ax.set_xlim3d([0.0, 35.0])
    ax.set_xlabel('X')
    ax.set_ylim3d([-10.0, 30.0])
    ax.set_ylabel('Y')
    ax.set_zlim3d([0.0, 10.0])
    ax.set_zlabel('Z')

    data = df[df['time'] == 0]
    graph = ax.scatter(data.x, data.y, data.z)

    ani = matplotlib.animation.FuncAnimation(fig, update_graph, len(landmarks_coordinates_3D[0]),  # fargs=(pos, lines),
                                             interval=60, blit=False)
    if save_path != '':
        writer_video = animation.PillowWriter(fps=60)
        ani.save(save_path, writer = writer_video)
    plt.show()

def smooth_curve(points_3D):
    x = []
    y = []
    z = []
    for i in range(len(points_3D)):
        x.append(points_3D[i][0])
        y.append(points_3D[i][1])
        z.append(points_3D[i][2])
    y_sf = savgol_filter(y, SMOOTH_FILTER_WINDOW, SMOOTH_FILTER_SMOOTHNESS)
    z_sf = savgol_filter(z, SMOOTH_FILTER_WINDOW, SMOOTH_FILTER_SMOOTHNESS)

    new_points_3D = []
    for i in range(len(points_3D)):
        new_points_3D.append([x[i], y_sf[i], z_sf[i]])

    return new_points_3D


def fill_empty_data(data):
    for i in range(len(data)):
        df = pd.DataFrame(data[i])
        data[i] = df.interpolate()
    return data


def points_null(p1, p2):
    if (p1[0] == None) or (p2[0] == None):
        return True
    return False
