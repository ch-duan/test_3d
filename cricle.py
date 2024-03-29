import cv2
import numpy as np
import time
from math import cos, sin, pi


# 回调函数，用于滑动条，这里不做任何操作
def nothing(x):
    pass

def find_circle_by_calipers(image, center, radius, num_points):
    # Prepare angles for calipers
    angles = np.linspace(0, 2 * pi, num_points)

    # List to hold edge points
    edge_points = []

    for angle in angles:
        # Start and end points for the calipers
        x1 = int(center[0] + radius * cos(angle))
        y1 = int(center[1] + radius * sin(angle))
        x2 = int(center[0] - radius * cos(angle))
        y2 = int(center[1] - radius * sin(angle))

        # Create a line iterator for the caliper line
        line_iter = cv2.LineIterator(image, (x1, y1), (x2, y2), connectivity=8)

        # Iterate over the line and find the edge point
        max_gradient = -1
        edge_point = None
        for pos in line_iter:
            # Get the pixel values along the line
            val = image[pos[1], pos[0]]
            if line_iter.pos() > 0:
                prev_val = image[line_iter.prevPos()[1], line_iter.prevPos()[0]]
                # Compute the gradient
                gradient = abs(val - prev_val)
                if gradient > max_gradient:
                    max_gradient = gradient
                    edge_point = (pos[0], pos[1])
        
        if edge_point is not None:
            edge_points.append(edge_point)

    # Now we have the edge points, we can fit a circle to these points
    # Fitting a circle is a more complex problem, you can use cv2.minEnclosingCircle() for an approximation
    # or implement a more accurate circle fitting algorithm if precision is important
    if len(edge_points) >= 3:
        circle_center, circle_radius = cv2.minEnclosingCircle(np.array(edge_points))
        return True, (int(circle_center[0]), int(circle_center[1])), int(circle_radius), edge_points
    else:
        return False, None, None, None


# def calculate_hexagon_vertices(center, radius):
#     # 定义60度的角度增量
#     angle_deg = 60
#     angle_rad = np.deg2rad(angle_deg)

#     # 计算六边形的顶点
#     vertices = []
#     for i in range(6):
#         angle_current = angle_rad * i
#         x = int(center[0] + radius * np.cos(angle_current))
#         y = int(center[1] + radius * np.sin(angle_current))
#         vertices.append((x, y))
#     return vertices


def calculate_hexagon_vertices(center, radius, rotation_deg=-16):
    # 将偏移角度从度转换为弧度
    rotation_rad = np.deg2rad(rotation_deg)
    # 定义60度的角度增量
    angle_deg = 60
    angle_rad = np.deg2rad(angle_deg)

    # 计算六边形的顶点
    vertices = []
    for i in range(6):
        # 每个顶点相对于中心点的角度（包括偏移）
        angle_current = rotation_rad + angle_rad * i
        x = int(center[0] + radius * np.cos(angle_current))
        y = int(center[1] + radius * np.sin(angle_current))
        vertices.append((x, y))
    return vertices


def average_center(circles):
    sum_x = 0
    sum_y = 0
    for i in circles[0, :]:
        sum_x += i[0]
        sum_y += i[1]
    ret = [int(sum_x / len(circles[0])), int(sum_y / len(circles[0]))]
    return ret


# 全局变量用于存储圆心和绘制的直线的端点
center = (-1, -1)
point1 = (-1, -1)
point2 = (-1, -1)


# 鼠标回调函数
def draw_line(event, x, y, flags, param):
    global center, point1, point2

    if event == cv2.EVENT_LBUTTONDOWN:
        center = (x, y)  # 假设每次左键按下的点是圆心
        point1 = center  # 开始画线的点

    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        point2 = (x, y)  # 当鼠标移动时更新直线的终点

    elif event == cv2.EVENT_LBUTTONUP:
        point2 = (x, y)  # 最终确定直线的终点

        # 计算直线的长度
        line_length = np.sqrt(
            (point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2
        )
        print(f"Line length: {line_length}")


# 创建一个窗口
cv2.namedWindow("Image")
# minDist = 50
# param1 = 54
# param2 = 29
# min_radius = 50
# max_radius = 64
# 创建滑动条，用于调节霍夫圆变换的参数
cv2.createTrackbar("minDist", "Image", 50, 100, nothing)
cv2.createTrackbar("Param1", "Image", 54, 100, nothing)
cv2.createTrackbar("Param2", "Image", 29, 100, nothing)
cv2.createTrackbar("Min Radius", "Image", 50, 300, nothing)
cv2.createTrackbar("Max Radius", "Image", 64, 300, nothing)

# cv2.setMouseCallback("Image", draw_line)


# 读取图像
image = cv2.imread("img2/Blue Ring 11.bmp")
image = cv2.resize(image, (0, 0), fx=0.35, fy=0.35)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# threshold = cv2.adaptiveThreshold(
#     gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 5
# )  # 自动阈值二值化

# cv2.imshow("Image", threshold)
# cv2.waitKey(0)

while True:
    # 获取滑动条的当前位置
    minDist = cv2.getTrackbarPos("minDist", "Image")
    param1 = cv2.getTrackbarPos("Param1", "Image")
    param2 = cv2.getTrackbarPos("Param2", "Image")
    min_radius = cv2.getTrackbarPos("Min Radius", "Image")
    max_radius = cv2.getTrackbarPos("Max Radius", "Image")
    # # 边缘检测
    # edges = cv2.Canny(gray, 50, 200, 3)
    # # 显示结果
    # cv2.imshow("Edges", edges)
    # cv2.waitKey(0)

    # # 找圆
    # # 轮廓检测

    # cv2.imshow("numpy_img", numpy_img)
    # cv2.waitKey(0)
    # 使用霍夫变换检测圆
    circles = cv2.HoughCircles(
        threshold,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    # 复制原图像
    displayed_image = image.copy()

    # 如果检测到圆，绘制圆
    if circles is not None:
        circles = np.uint16(np.around(circles))
        center = average_center(circles)
         # 将(x, y)坐标和半径转换为整数
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            # 使用霍夫圆检测的结果来调用find_circle_by_calipers函数
            found, center, radius, edge_points = find_circle_by_calipers(image, (x, y), r, 36)

            # 绘制结果
            if found:
                for point in edge_points:
                    cv2.circle(displayed_image, point, 1, (0, 0, 255), -1)
                cv2.circle(displayed_image, center, radius, (0, 255, 0), 1)
        # # cv2.circle(displayed_image, center, 5, (158, 52, 235), 3)
        # for i in circles[0, :]:
        #     # 绘制圆心
        #     cv2.circle(displayed_image, (i[0], i[1]), 1, (0, 100, 100), 3)
        #     # 绘制圆轮廓
        #     cv2.circle(displayed_image, (i[0], i[1]), i[2], (255, 0, 255), 2)

    # 计算六边形的顶点
    # 461
    hexagon_vertices = calculate_hexagon_vertices(center, 540)

    # 将顶点列表转换为NumPy数组
    hexagon = np.array(hexagon_vertices, np.int32)
    hexagon = hexagon.reshape((-1, 1, 2))

    # 绘制六边形
    cv2.polylines(displayed_image, [hexagon], True, (158, 52, 235), 2)
    
    

    # # 显示图像
    # if center[0] != -1:
    #     cv2.circle(displayed_image, center, 3, (0, 255, 0), -1)
    # if point1[0] != -1 and point2[0] != -1:
    #     cv2.line(displayed_image, point1, point2, (255, 0, 0), 2)

    cv2.imshow("Image", displayed_image)

    # 等待按键退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    time.sleep(0.5)

# 清除所有窗口
cv2.destroyAllWindows()
