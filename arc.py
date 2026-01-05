import cv2
import numpy as np
import math
import sys

SCENE_FILE = 'scene.jpg'
MODEL_IMG_FILE = 'model.jpg'
OBJ_FILE = 'mod.obj'
OUTPUT_FILE = 'output.png'
MIN_MATCHES = 30
SCALE = 4.0
FOCAL_LENGTH_RATIO = 1.0

def hex_to_rgb(hex_color):
    if isinstance(hex_color, str):
        hex_color = hex_color.lstrip('#')
    else:
        raise TypeError("hex_color must be a string")
    if len(hex_color) != 6:
        raise ValueError("hex_color must be in RRGGBB format")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)

class SimpleOBJ:
    def __init__(self, filename):
        self.vertices = []
        self.faces = []
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    x, y, z = map(float, parts[1:4])
                    self.vertices.append([x, y, z])
                elif line.startswith('f '):
                    parts = line.strip().split()[1:]
                    face_idx = []
                    for p in parts:
                        idx = p.split('/')[0]
                        face_idx.append(int(idx))
                    self.faces.append((face_idx, None))

def projection_matrix(camera_parameters, homography):
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]

    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l

    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)

    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)

    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def render(scene_img, obj, projection, reference_image, color=False, scale=SCALE):
    vertices = np.array(obj.vertices)
    scale_matrix = np.eye(3) * scale
    h, w = reference_image.shape[:2]

    out = scene_img.copy()

    for face in obj.faces:
        face_vertices = face[0]
        pts = np.array([vertices[idx - 1] for idx in face_vertices])
        pts = np.dot(pts, scale_matrix)
        pts = np.array([[p[0] + w / 2.0, p[1] + h / 2.0, p[2]] for p in pts])
        dst = cv2.perspectiveTransform(pts.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst.reshape(-1, 2))
        if color is False:
            cv2.fillConvexPoly(out, imgpts, (137, 27, 211))
        else:
            col = hex_to_rgb(face[-1])
            col = col[::-1]
            cv2.fillConvexPoly(out, imgpts, col)
    return out

def main():
    scene_color = cv2.imread(SCENE_FILE, cv2.IMREAD_COLOR)
    if scene_color is None:
        print("Could not open scene image:", SCENE_FILE)
        return
    scene_gray = cv2.cvtColor(scene_color, cv2.COLOR_BGR2GRAY)

    model_img = cv2.imread(MODEL_IMG_FILE, cv2.IMREAD_GRAYSCALE)
    if model_img is None:
        print("Could not open model image:", MODEL_IMG_FILE)
        return

    orb = cv2.ORB_create()
    kp_model, des_model = orb.detectAndCompute(model_img, None)
    kp_scene, des_scene = orb.detectAndCompute(scene_gray, None)
    if des_model is None or des_scene is None:
        print("No descriptors found in model or scene. Check images.")
        return

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_model, des_scene)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < MIN_MATCHES:
        print(f"Not enough matches have been found - {len(matches)}/{MIN_MATCHES}")
        return

    good_matches = matches[:MIN_MATCHES]

    match_img = cv2.drawMatches(model_img, kp_model, scene_gray, kp_scene,
                                good_matches, None, flags=2)
    cv2.imshow('Matches', match_img)
    cv2.waitKey(500)

    src_pts = np.float32([kp_model[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is None or int(mask.sum()) < 10:
        print("Homography failed or not enough inliers to reliably locate the object")
        return

    h_model, w_model = model_img.shape
    pts = np.float32([[0, 0], [0, h_model - 1], [w_model - 1, h_model - 1], [w_model - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    scene_with_box = cv2.polylines(scene_color.copy(), [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('Detected area', scene_with_box)
    cv2.waitKey(300)

    h_scene, w_scene = scene_gray.shape
    f = FOCAL_LENGTH_RATIO * max(w_scene, h_scene)
    K = np.array([[f, 0, w_scene / 2.0],
                  [0, f, h_scene / 2.0],
                  [0, 0, 1]], dtype=float)

    projection = projection_matrix(K, M)

    try:
        obj = SimpleOBJ(OBJ_FILE)
    except Exception as e:
        print("Failed to load OBJ:", e)
        return

    if len(obj.vertices) == 0 or len(obj.faces) == 0:
        print("OBJ has no vertices or faces.")
        return

    result = render(scene_color, obj, projection, model_img, color=False, scale=SCALE)

    cv2.imwrite(OUTPUT_FILE, result)
    print(f"Saved output to {OUTPUT_FILE}")
    cv2.imshow('Rendered result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()