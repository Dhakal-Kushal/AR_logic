import argparse
import cv2
import numpy as np
import math
import os
from objloader_simple import *

# ---------------- CONFIG ----------------
MIN_MATCHES = 6                  # Lowered to allow rendering with fewer inliers
DEFAULT_COLOR = (0, 255, 0)     # Green so the OBJ is visible
FRAME_SKIP = 2                   # Optional, keeps some frames for performance
# ----------------------------------------

def main():
    homography = None

    # Camera parameters (approximate)
    camera_parameters = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0,   0,   1]
    ], dtype=np.float32)

    # ORB detector
    orb = cv2.ORB_create(nfeatures=1000)

    # BFMatcher (KNN, no crossCheck)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Load reference image
    dir_name = os.getcwd()
    model = cv2.imread(os.path.join(dir_name, 'model.jpg'), cv2.IMREAD_GRAYSCALE)
    if model is None:
        raise RuntimeError("Failed to load model.jpg")

    kp_model, des_model = orb.detectAndCompute(model, None)
    if des_model is None:
        raise RuntimeError("No features found in model image")

    # Load OBJ model
    obj = OBJ(os.path.join(dir_name, 'FinalBaseMesh.obj'), swapyz=True)

    # Video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = orb.detectAndCompute(gray, None)
        if des_frame is None:
            cv2.imshow("frame", frame)
            continue

        # -------- KNN MATCH + RATIO TEST --------
        matches = bf.knnMatch(des_model, des_frame, k=2)
        good = []
        for m_n in matches:
            if len(m_n) != 2:    # skip if less than 2 matches
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)


        if len(good) < MIN_MATCHES:
            cv2.imshow("frame", frame)
            continue

        print(f"Good matches: {len(good)}")

        # -------- HOMOGRAPHY --------
        src_pts = np.float32([kp_model[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0
        print(f"Inliers: {inliers}")

        # Accept homography if at least MIN_MATCHES inliers
        if H is None or inliers < MIN_MATCHES:
            cv2.imshow("frame", frame)
            continue

        # -------- OPTIONAL RECTANGLE --------
        if args.rectangle:
            h, w = model.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, H)
            frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        # -------- RENDER 3D MODEL --------
        print("RENDERING MODEL")   # Debug: confirm rendering
        projection = projection_matrix(camera_parameters, H)
        frame = render(frame, obj, projection, model)

        # -------- OPTIONAL MATCH VISUALIZATION --------
        if args.matches:
            frame = cv2.drawMatches(
                model, kp_model,
                frame, kp_frame,
                good[:10], None,
                flags=2
            )

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------- RENDER ----------------
def render(img, obj, projection, model, color=False):
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 1.5   # Adjusted scale to be visible on card
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[v - 1] for v in face_vertices])
        points = np.dot(points, scale_matrix)

        # Center on reference surface
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)

        if not color:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            c = hex_to_rgb(face[-1])[::-1]
            cv2.fillConvexPoly(img, imgpts, c)

    return img


def projection_matrix(camera_parameters, homography):
    homography = -homography
    rot_trans = np.dot(np.linalg.inv(camera_parameters), homography)

    col_1 = rot_trans[:, 0]
    col_2 = rot_trans[:, 1]
    col_3 = rot_trans[:, 2]

    l = math.sqrt(np.linalg.norm(col_1) * np.linalg.norm(col_2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l

    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)

    rot_1 = (c / np.linalg.norm(c) + d / np.linalg.norm(d)) / math.sqrt(2)
    rot_2 = (c / np.linalg.norm(c) - d / np.linalg.norm(d)) / math.sqrt(2)
    rot_3 = np.cross(rot_1, rot_2)

    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


# ---------------- CLI ----------------
parser = argparse.ArgumentParser(description='Augmented reality application')
parser.add_argument('-r', '--rectangle', action='store_true')
parser.add_argument('-ma', '--matches', action='store_true')
args = parser.parse_args()

if __name__ == '__main__':
    main()
