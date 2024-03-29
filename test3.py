import cv2
import numpy as np
import cv2
import numpy as np
import time
params = cv2.SimpleBlobDetector_Params()
# 设置阈值
params.minThreshold = 50
params.maxThreshold = 200
# 设置选择区域
params.filterByArea = True
params.minArea = 100
# 设置圆度
params.filterByCircularity = True
params.minCircularity = 0.1
# 设置凸度
params.filterByConvexity = True
params.minConvexity = 0.1
# 这种惯性比
params.filterByInertia = True
params.minInertiaRatio = 0.01
detector = cv2.SimpleBlobDetector_create(params)
im = cv2.imread("img/4.bmp", cv2.COLOR_BGR2GRAY)
img2 = cv2.resize(im, (0,0), fx=0.5, fy=0.5)
cv2.imshow("img",img2)
cv2.waitKey(0)
keypoints = detector.detect(im)
with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]),
(0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
with_keypoints = cv2.resize(with_keypoints, (0,0), fx=0.5, fy=0.5)
cv2.imshow("Keypoints", with_keypoints)
cv2.waitKey(0)