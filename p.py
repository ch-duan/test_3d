import cv2
import cv2 as cv
import matplotlib.pyplot as plt
 
# 1 图像读取
img = cv.imread('img2/Blue Ring 11.bmp')
img=cv.resize(img,(0,0),fx=0.45,fy=0.45)
img1 = img.copy()
imgray = cv.cvtColor(img, cv2.COLOR_BGR2GRAY)
threshold = cv2.adaptiveThreshold(
    imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 5
)  # 自动阈值二值化
# 2 边缘检测
canny = cv.Canny(threshold, 127, 255, 0)
# 3 轮廓提取
contours, hierarchy = cv.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# 4 将轮廓绘制在图像上
img = cv.drawContours(img, contours, 1, (255, 0, 0), 2)
cv.imshow('img', img)
cv.waitKey(0)
# 5 凸包检测
hulls = []
for cnt in contours:
    # 寻找凸包使用cv2.convexHull(contour)
    hull = cv.convexHull(cnt)
    hulls.append(hull)
draw_hulls = cv.drawContours(img1, hulls, -1, (0, 255, 0), 2)
 
# 5 图像显示
plt.figure(figsize=(10, 8), dpi=100)
plt.subplot(121), plt.imshow(img[:, :, ::-1]), plt.title('轮廓检测结果')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(draw_hulls[:, :, ::-1]), plt.title('凸包结果')
plt.xticks([]), plt.yticks([])
plt.show()



# import cv2 
# import numpy as np

# img = cv2.imread('img/4.jpg')
# img=cv2.resize(img,(0,0),fx=0.45,fy=0.45)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# ret,binary = cv2.threshold(gray,60,255,0)#阈值处理
# # contours,hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#查找轮廓
# contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours))
# x = 0
# for i in range(len(contours)):
#     area = cv2.contourArea(contours[i])
#     if area>10000:
#         print(area)
#         x = i
# cnt = contours[x]
# img1 = img.copy()
# approx1 = cv2.approxPolyDP(cnt,3,True)#拟合精确度
# img1  =cv2.polylines(img1,[approx1],True,(255,255,0),2)
# cv2.imshow('approxPolyDP1',img1)
# cv2.waitKey(0)
# img2 = img.copy()
# approx2 = cv2.approxPolyDP(cnt,5,True)#拟合精确度
# img2  =cv2.polylines(img2,[approx2],True,(255,255,0),2)
# cv2.imshow('approxPolyDP2',img2)
# cv2.waitKey(0)
# img3 = img.copy()
# approx3 = cv2.approxPolyDP(cnt,7,True)#拟合精确度
# img3  =cv2.polylines(img3,[approx3],True,(255,255,0),2)
# cv2.imshow('approxPolyDP3',img3)


# print(len(approx1))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
