import cv2
import numpy as np
import math
from collections import deque

CAMERA_INDEX = 0
MODEL_IMG_FILE = 'models/ref.jpg'
MODEL_FILE = '3dObject.obj'

MIN_MATCHES = 10
MIN_INLIERS = 8
SCALE = 200.0
SMOOTHING_FRAMES = 3

class SimpleOBJ:
    def __init__(self, filename):
        self.vertices = []
        self.faces = []
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    _, x, y, z = line.split()
                    self.vertices.append([float(x), float(y), float(z)])
                elif line.startswith('f '):
                    parts = line.strip().split()[1:]
                    self.faces.append(([int(p.split('/')[0]) for p in parts], None))

def projection_matrix(camera_parameters, homography):
    """Compute 3D projection matrix, from camera calibration matrix and homography"""
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

def render(img, obj, projection, model, scale=SCALE):
    """Render loaded obj model into current video frame"""
    vertices = obj.vertices
    scale_matrix = np.eye(3) * scale
    h, w = model.shape
    
    out = img.copy()
    
    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        
        cv2.fillConvexPoly(out, imgpts, (137, 27, 211))
        cv2.polylines(out, [imgpts], True, (255, 255, 0), 2)
    
    return out

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open camera")
        return

    model_img = cv2.imread(MODEL_IMG_FILE, cv2.IMREAD_GRAYSCALE)
    if model_img is None:
        print("Failed to load model image")
        return

    print(f"Model image size: {model_img.shape}")

    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)

    try:
        sift = cv2.SIFT_create()
        kp_model, des_model = sift.detectAndCompute(model_img, None)
        use_sift = True
        print("Using SIFT detector")
    except:
        orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8)
        kp_model, des_model = orb.detectAndCompute(model_img, None)
        use_sift = False
        print("Using ORB detector")
    
    print(f"Model keypoints found: {len(kp_model)}")

    obj = SimpleOBJ(MODEL_FILE)
    print(f"Model loaded: {len(obj.vertices)} vertices, {len(obj.faces)} faces")

    if use_sift:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matcher = bf

    print("Press ESC to exit")
    frame_count = 0
    homography_buffer = deque(maxlen=SMOOTHING_FRAMES)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if use_sift:
            kp_scene, des_scene = sift.detectAndCompute(gray, None)
        else:
            kp_scene, des_scene = orb.detectAndCompute(gray, None)

        if des_scene is None or len(kp_scene) < 2:
            cv2.putText(frame, "No features detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("AR", frame)
            if cv2.waitKey(1) == 27:
                break
            continue

        try:
            if use_sift:
                matches = matcher.knnMatch(des_model, des_scene, k=2)
                good = []
                for m_n in matches:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < 0.7 * n.distance:
                            good.append(m)
            else:
                matches = matcher.match(des_model, des_scene)
                matches = sorted(matches, key=lambda x: x.distance)
                good = matches[:min(50, len(matches))]
        except cv2.error:
            cv2.putText(frame, "Too few features", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("AR", frame)
            if cv2.waitKey(1) == 27:
                break
            continue

        frame_count += 1
        
        if len(good) < MIN_MATCHES:
            cv2.putText(frame, f"Matches: {len(good)}/{MIN_MATCHES}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("AR", frame)
            if cv2.waitKey(1) == 27:
                break
            continue

        src_pts = np.float32([kp_model[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            inliers = int(mask.sum())
            
            if inliers >= MIN_INLIERS:
                homography_buffer.append(M)
                if len(homography_buffer) >= 2:
                    M_smooth = np.mean(homography_buffer, axis=0)
                else:
                    M_smooth = M
                
                try:
                    projection = projection_matrix(camera_parameters, M_smooth)
                    frame = render(frame, obj, projection, model_img)
                    
                    cv2.putText(frame, f"Tracking! ({inliers} inliers)", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except Exception as e:
                    if frame_count % 30 == 0:
                        print(f"Render error: {e}")

        cv2.imshow("AR", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()