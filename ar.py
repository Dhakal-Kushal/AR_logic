import cv2
import numpy as np

img = cv2.imread('scene.jpg', 0)

# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img, None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location, not size and orientation
img2 = cv2.drawKeypoints(img, kp, img, color=(0,255,0), flags=0)
cv2.imshow('keypoints', img2)
cv2.waitKey(0)

MIN_MATCHES = 15
cap = cv2.imread('scene.jpg', 0)
model = cv2.imread('model.jpg', 0)
# ORB keypoint detector
orb = cv2.ORB_create()
# create brute force  matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Compute model keypoints and its descriptors
kp_model, des_model = orb.detectAndCompute(model, None)
# Compute scene keypoints and its descriptors
kp_frame, des_frame = orb.detectAndCompute(cap, None)
# Match frame descriptors with model descriptors
matches = bf.match(des_model, des_frame)
# Sort them in the order of their distance
matches = sorted(matches, key=lambda x: x.distance)

if len(matches) > MIN_MATCHES:
    # Use only the best matches for homography and drawing
    good_matches = matches[:MIN_MATCHES]

    # draw first MIN_MATCHES matches on a separate image (do not overwrite `cap`)
    match_img = cv2.drawMatches(model, kp_model, cap, kp_frame,
                               good_matches, None, flags=2)
    cv2.imshow('frame_matches', match_img)
    cv2.waitKey(0)

    # differenciate between source points and destination points using only good_matches
    src_pts = np.float32([kp_model[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # compute Homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)

    # Validate homography and number of inliers
    if M is None or int(mask.sum()) < 10:
        print("Homography failed or not enough inliers to reliably locate the object")
    else:
        # Draw a rectangle that marks the found model in the frame
        h, w = model.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # project corners into frame
        dst = cv2.perspectiveTransform(pts, M)
        # connect them with lines on a copy of the original scene image
        scene_with_box = cv2.polylines(cap.copy(), [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        cv2.imshow('frame_detected', scene_with_box)
        cv2.waitKey(0)
else:
    print(f"Not enough matches have been found -{len(matches)}/{MIN_MATCHES}")
