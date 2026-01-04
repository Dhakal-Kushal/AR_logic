# augR.py
import argparse
import cv2
import numpy as np
import math
import os
import sys
from objloader_simple import *

MIN_MATCHES = 10
DEFAULT_COLOR = (0, 0, 0)

def main():
    """
    This function loads the target surface image and runs the AR loop.
    """
    homography = None 
    # camera parameters (tune as needed)
    camera_parameters = np.array([
    [600, 0, 320],
    [0, 600, 240],
    [0,   0,   1]
    ])

    # create ORB keypoint detector (increase nfeatures if needed)
    orb = cv2.ORB_create(
    nfeatures=2000,
    scaleFactor=1.2,
    nlevels=8)

    # BFMatcher, we'll use knnMatch (crossCheck=False)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    dir_name = os.getcwd()
    model = cv2.imread(os.path.join(dir_name, 'model.jpg'), 0)
    if model is None:
        print("Error: model.jpg not found or cannot be read.")
        return 1

    # Compute model keypoints and its descriptors
    kp_model, des_model = orb.detectAndCompute(model, None)
    if des_model is None or len(kp_model) == 0:
        print("Error: no keypoints/descriptors found in model.jpg")
        return 1

    # Load 3D model from OBJ file
    obj_path = os.path.join(dir_name, 'FinalBaseMesh.obj')
    if not os.path.exists(obj_path):
        print("Error: FinalBaseMesh.obj not found.")
        return 1
    obj = OBJ(obj_path, swapyz=True)

    # init video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam.")
        return 1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            break

        # convert frame to grayscale to match model processing (more consistent descriptors)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # find and draw the keypoints of the frame
        kp_frame, des_frame = orb.detectAndCompute(frame_gray, None)

        # Safety: if descriptors are missing (blank wall / finger over lens) skip matching
        if des_frame is None or len(kp_frame) == 0:
            # show the live frame with a status overlay and continue loop
            display = frame.copy()
            cv2.putText(display, "No descriptors in frame (low texture/occlusion)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('frame', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Use knnMatch + Lowe ratio test for more stable matches
        try:
            raw_matches = bf.knnMatch(des_model, des_frame, k=2)
        except Exception as e:
            # unexpected mismatch in descriptor types or sizes
            print("Matcher error:", e)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ratio test
        good_matches = []
        ratio_thresh = 0.85
        for m_n in raw_matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        # sort by distance (best matches first)
        good_matches = sorted(good_matches, key=lambda x: x.distance)

        # compute Homography if enough matches are found
        if len(good_matches) > MIN_MATCHES:
            src_pts = np.float32([kp_model[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if homography is not None and mask is not None:
                inliers = int(mask.sum())
                if inliers >= max(8, MIN_MATCHES // 2):
                    if args.rectangle:
                        h, w = model.shape
                        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, homography)
                        frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

                    # if a valid homography matrix was found render model on model plane
                    try:
                        projection = projection_matrix(camera_parameters, homography)
                        frame = render(frame, obj, projection, model, False)
                    except Exception as e:
                        # don't let a transient projection/render error crash the loop
                        print("Render/projection error:", e)

                    # optionally draw matches (only the top 10 good matches)
                    if args.matches:
                        # draw only good matches (top 10) - map indices to the original
                        frame = cv2.drawMatches(model, kp_model, frame, kp_frame, good_matches[:10], None, flags=2)

                else:
                    # not enough inlier matches after RANSAC
                    cv2.putText(frame, f"Inliers after RANSAC too few: {inliers}/{MIN_MATCHES}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Homography failed", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, f"Not enough good matches: {len(good_matches)}/{MIN_MATCHES}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Always show the current frame (so UI stays responsive) and allow quit
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0

def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices

    # FIX: define h, w BEFORE using them
    h, w = model.shape

    # Scale relative to card size
    scale = min(w, h) / 150.0
    scale_matrix = np.eye(3) * scale

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)

        # Center model on the reference image
        points = np.array([
            [p[0] + w / 2, p[1] + h / 2, p[2]]
            for p in points
        ])

        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)

        if color is False:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            color = hex_to_rgb(face[-1])[::-1]
            cv2.fillConvexPoly(img, imgpts, color)

    return img


def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

# Command line argument parsing
parser = argparse.ArgumentParser(description='Augmented reality application')
parser.add_argument('-r','--rectangle', help = 'draw rectangle delimiting target surface on frame', action = 'store_true')
parser.add_argument('-mk','--model_keypoints', help = 'draw model keypoints', action = 'store_true')
parser.add_argument('-fk','--frame_keypoints', help = 'draw frame keypoints', action = 'store_true')
parser.add_argument('-ma','--matches', help = 'draw matches between keypoints', action = 'store_true')

args = parser.parse_args()

if __name__ == '__main__':
    main()
