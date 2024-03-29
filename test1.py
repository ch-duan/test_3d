import cv2
import numpy as np
# 读取图像
img = cv2.imread('img/55.jpg')
#缩放一半
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
cv2.waitKey(0)
cv2.imshow('img', img)
# 灰度化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 边缘检测
edges = cv2.Canny(gray, 50,200,3)
# 显示结果
cv2.imshow('Edges', edges)
cv2.waitKey(0)

#找圆
# 轮廓检测
numpy_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 15)   # 自动阈值二值化


cv2.imshow('numpy_img', numpy_img)
cv2.waitKey(0)

circles = cv2.HoughCircles(numpy_img, cv2.HOUGH_GRADIENT, 1, 50,   param1=80, param2=60,  minRadius=50, maxRadius=130)

arr1 = np.zeros([0, 2], dtype=int)                      # 创建一个0行, 2列的空数组
if circles is not None:
    circles = np.uint16(np.around(circles))   # 4舍5入, 然后转为uint16
    for i in circles[0, :]:
        arr1 = np.append(arr1, (i[0], i[1]))            # arr1是圆心坐标的np数组
        # print(arr1)
        cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 255), 3)  # 轮廓
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 0), 6)     # 圆心

cv2.imshow('circle', img)
cv2.waitKey(0)

# contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# #绘制轮廓
# cv2.drawContours(img, contours, -1, (0, 255, 255), 2)
# #显示结果
# cv2.imshow('Contours', img)
# cv2.waitKey(0)

# 2. 轮廓检测：
# 读取图像
# img = cv2.imread('image.jpg')
# 灰度化
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 边缘检测
# edges = cv2.Canny(gray, 50, 150)
# 轮廓检测
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 绘制轮廓
cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
# 显示结果
cv2.imshow('Contours', img)
cv2.waitKey(0)

# 3. 直线检测：
# 读取图像
# img = cv2.imread('image.jpg')
# 灰度化
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 边缘检测
# edges = cv2.Canny(gray, 50, 150)
# 直线检测
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
# 绘制直线
for line in lines:
  rho, theta = line[0]
  a = np.cos(theta)
  b = np.sin(theta)
  x0 = a * rho
  y0 = b * rho
  x1 = int(x0 + 1000 * (-b))
  y1 = int(y0 + 1000 * (a))
  x2 = int(x0 - 1000 * (-b))
  y2 = int(y0 - 1000 * (a))
  cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
# 显示结果
cv2.imshow('Lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()