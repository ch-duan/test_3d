import math
import cv2
import numpy as np
import sympy

# 凸包检测和凸缺陷
import cv2 as cv


def getDist_P2P(Point0, PointA):
    distance = math.pow((Point0[0] - PointA[0]), 2) + math.pow(
        (Point0[1] - PointA[1]), 2
    )
    distance = math.sqrt(distance)
    return distance


def myapproxPolyDP(contours, minepsilon=1, maxepsilon=20, sides=6):
    rect1 = cv2.approxPolyDP(contours, minepsilon, True)
    rect2 = cv2.approxPolyDP(contours, maxepsilon, True)
    print(len(rect1), len(rect2))
    if len(rect1) > sides and len(rect2) > sides:
        return contours, False
    if len(rect1) < sides and len(rect2) < sides:
        return contours, False
    else:
        if len(rect1) == sides:
            return rect1[:sides], True
        elif len(rect2) == sides:
            return rect2[:sides], True
        else:
            midepsilon = (minepsilon + maxepsilon) / 2.0
            rect3 = cv2.approxPolyDP(contours, midepsilon, True)
            if len(rect3) == sides:
                return rect3[:sides], True
            elif len(rect3) < sides:
                return myapproxPolyDP(contours, minepsilon, midepsilon)
            else:
                return myapproxPolyDP(contours, midepsilon, maxepsilon)


def appx_best_fit_ngon(mask_cv2, n: int = 6) -> list[(int, int)]:
    # convex hull of the input mask
    mask_cv2_gray = cv2.cvtColor(mask_cv2, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(
        mask_cv2_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    hull = cv2.convexHull(contours[0])
    hull = np.array(hull).reshape((len(hull), 2))

    # to sympy land
    hull = [sympy.Point(*pt) for pt in hull]

    # run until we cut down to n vertices
    while len(hull) > n:
        best_candidate = None

        # for all edges in hull ( <edge_idx_1>, <edge_idx_2> ) ->
        for edge_idx_1 in range(len(hull)):
            edge_idx_2 = (edge_idx_1 + 1) % len(hull)

            adj_idx_1 = (edge_idx_1 - 1) % len(hull)
            adj_idx_2 = (edge_idx_1 + 2) % len(hull)

            edge_pt_1 = sympy.Point(*hull[edge_idx_1])
            edge_pt_2 = sympy.Point(*hull[edge_idx_2])
            adj_pt_1 = sympy.Point(*hull[adj_idx_1])
            adj_pt_2 = sympy.Point(*hull[adj_idx_2])

            subpoly = sympy.Polygon(adj_pt_1, edge_pt_1, edge_pt_2, adj_pt_2)
            angle1 = subpoly.angles[edge_pt_1]
            angle2 = subpoly.angles[edge_pt_2]

            # we need to first make sure that the sum of the interior angles the edge
            # makes with the two adjacent edges is more than 180°
            if sympy.N(angle1 + angle2) <= sympy.pi:
                continue

            # find the new vertex if we delete this edge
            adj_edge_1 = sympy.Line(adj_pt_1, edge_pt_1)
            adj_edge_2 = sympy.Line(edge_pt_2, adj_pt_2)
            intersect = adj_edge_1.intersection(adj_edge_2)[0]

            # the area of the triangle we'll be adding
            area = sympy.N(sympy.Triangle(edge_pt_1, intersect, edge_pt_2).area)
            # should be the lowest
            if best_candidate and best_candidate[1] < area:
                continue

            # delete the edge and add the intersection of adjacent edges to the hull
            better_hull = list(hull)
            better_hull[edge_idx_1] = intersect
            del better_hull[edge_idx_2]
            best_candidate = (better_hull, area)

        if not best_candidate:
            raise ValueError("Could not find the best fit n-gon!")

        hull = best_candidate[0]

    # back to python land
    hull = [(int(x), int(y)) for x, y in hull]

    return hull


class hexagon_hull:
    def __init__(self):
        self.start = tuple()
        self.end = tuple()
        self.lenght = 0


def intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception("lines do not intersect")

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def find_hexagon(img, contours):
    for c in range(len(contours)):
        # area=cv2.contourArea(contours[c])
        # print("area",area)
        # x,y,w,h=cv2.boundingRect(contours[c])

        # mm=cv2.moments(contours[c])
        # print("mm",mm)
        # cx=mm["m10"]/mm["m00"]
        # cy=mm["m01"]/mm["m00"]
        # cv2.circle(img, (int(cx),int(cy)), 5, (0, 255, 0), -1)
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

        # 是否为凸包
        ret = cv.isContourConvex(contours[c])
        # 凸缺陷
        # 凸包检测，returnPoints为false的是返回与凸包点对应的轮廓上的点对应的index
        hull = cv.convexHull(contours[c], returnPoints=False)
        defects = cv.convexityDefects(contours[c], hull)
        hexagon = [hexagon_hull() for i in range(defects.shape[0])]
        for j in range(defects.shape[0]):
            s, e, f, d = defects[j, 0]
            start = tuple(contours[c][s][0])
            end = tuple(contours[c][e][0])
            far = tuple(contours[c][f][0])
            hexagon[j].start = start
            hexagon[j].end = end
            hexagon[j].lenght = getDist_P2P(start, end)

            # 用红色连接凸缺陷的起始点和终止点
            # cv.line(img, start, end, (0, 0, 225), 2)
            # # 用蓝色最远点画一个圆圈
            # cv.circle(img, far, 5, (225, 0, 0), -1)
            cnt = contours[c]
        return hexagon


def line_length(line):
    # 计算线段的长度
    p1, p2 = line
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def normalize_line(line):
    # 确保线段的起点总是小于终点
    (x1, y1), (x2, y2) = line
    return ((x1, y1), (x2, y2)) if (x1, y1) < (x2, y2) else ((x2, y2), (x1, y1))


def merge_lines(lines, angle_threshold=np.pi / 18, min_lines=12, length_threshold=10):
    # 计算线段的角度
    angles = [
        np.arctan2(line[1][1] - line[0][1], line[1][0] - line[0][0]) for line in lines
    ]
    # 将线段按角度分组
    grouped_lines = {}
    for angle, line in zip(angles, lines):
        matched = False
        for group_angle in grouped_lines:
            if abs(group_angle - angle) < angle_threshold:
                grouped_lines[group_angle].append(line)
                matched = True
                break
        if not matched:
            grouped_lines[angle] = [line]

    # 合并每个角度组内的线段
    merged_lines = []
    for group_lines in grouped_lines.values():
        # 计算所有线段的端点的凸包
        all_points = np.vstack(group_lines)
        hull_points = cv2.convexHull(all_points)
        # 对凸包的点进行多边形近似
        epsilon = 0.02 * cv2.arcLength(hull_points, True)
        approx = cv2.approxPolyDP(hull_points, epsilon, True)
        # 近似点可以被视为合并后的线段的端点
        temp_lines = [
            (tuple(approx[i][0]), tuple(approx[(i + 1) % len(approx)][0]))
            for i in range(len(approx))
        ]
        # 去除长度小于阈值的线段
        temp_lines = [
            line for line in temp_lines if line_length(line) >= length_threshold
        ]
        merged_lines.extend(temp_lines)

    # 可能需要进一步步骤来减少线段数量到min_lines
    # ...
    # 规范化合并后的线段
    normalized_lines = [normalize_line(line) for line in merged_lines]

    # 去除重复的线段
    unique_lines = []
    for line in normalized_lines:
        if line not in unique_lines:
            unique_lines.append(line)
    return unique_lines


def detect_hull():
    # 读取图像
    img = cv.imread("img/15.bmp")
    img = cv.resize(img, (0, 0), fx=0.3, fy=0.3)
    # 转换为灰度图像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)
    cv.waitKey(0)
    # 二值化
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    cv.waitKey(0)
    # 获取结构元素
    k = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # 开操作
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, k)
    # 轮廓发现
    contours, hierarchy = cv.findContours(
        binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    # 在原图上绘制轮廓，以方便和凸包对比，发现凸缺陷
    cv.drawContours(img, contours, -1, (0, 225, 0), 3)
    print("contours is ", len(contours))
    cv.imshow("result", img)
    cv.waitKey(0)
    for c in range(len(contours)):
        # 是否为凸包
        ret = cv.isContourConvex(contours[c])
        print("ret is ", ret)
        if ret == False:
            continue
        # 凸缺陷
        # 凸包检测，returnPoints为false的是返回与凸包点对应的轮廓上的点对应的index
        hull = cv.convexHull(contours[c])
        cv.drawContours(img, [hull], -1, (0, 0, 255), 2)
        cv.imshow("result", img)
        cv.waitKey(0)
        hull_lines = []

        # 遍历凸包中的点
        for i in range(len(hull)):
            # 当前点
            pt1 = tuple(hull[i][0])
            # 下一个点，如果当前点是最后一个点，则下一个点是第一个点
            pt2 = tuple(hull[(i + 1) % len(hull)][0])
            # 将当前点和下一个点作为线段的端点
            hull_lines.append((pt1, pt2))

        m_lines = merge_lines(
            hull_lines, angle_threshold=np.pi / 18, min_lines=6, length_threshold=100
        )
        print(len(m_lines))
        print(m_lines)
        hexagon = [hexagon_hull() for i in range(len(m_lines))]
        for i in range(len(m_lines)):
            print(m_lines[i])
            hexagon[i].start = m_lines[i][0]
            hexagon[i].end = m_lines[i][1]
            hexagon[i].lenght = getDist_P2P(m_lines[i][0], m_lines[i][1])
            print(hexagon[i].start, hexagon[i].end, hexagon[i].lenght)
            # x, y = intersection(line1, line2)
            # cv.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
            # cv.imshow("result", img)
            # cv.waitKey(0)

            # cv.line(img, m_lines[i][0], m_lines[i][1], (0, 0, 255), 2)
            # cv2.imshow("img", img)
            # cv2.waitKey(0)
        intersection_points = []
        for i in range(len(m_lines)):
            # sorted_hexagons[0]和sorted_hexagons[1]的交点
            s = i
            e = (i + 1) % len(m_lines)
            line1 = [hexagon[s].start, hexagon[s].end]
            line2 = [hexagon[e].start, hexagon[e].end]
            # cv.line(img, hexagon[s].start, hexagon[s].end, (0, 0, 225), 2)
            x, y = intersection(line1, line2)
            intersection_points.append((int(x), int(y)))
            cv.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv.imshow("result", img)
            cv.waitKey(0)

        for i in range(len(intersection_points)):

            s = i
            e = (i + 1) % len(intersection_points)
            print(getDist_P2P(intersection_points[s], intersection_points[e]))
            cv.line(
                img, intersection_points[s], intersection_points[e], (0, 255, 225), 2
            )
            cv.imshow("result", img)
            cv.waitKey(0)

        # # 绘制凸包的边界线
        # # cv2.polylines(image, [hull], True, (255, 255, 255), 2)

        # # 函数用来计算两点之间的角度
        # def calculate_angle(pt1, pt2):
        #     return np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])

        # # 存储凸包边界线段的角度
        # angles = []

        # # 计算凸包边界线段的角度
        # for i in range(len(hull)):
        #     pt1 = hull[i][0]
        #     pt2 = hull[(i + 1) % len(hull)][0]  # 循环到第一个点
        #     angle = calculate_angle(pt1, pt2)
        #     angles.append(angle)

        # # 判断和第一条线段方向一样的线段
        # # 你可以通过调整阈值来控制"方向一样"的判断标准
        # angle_threshold = np.pi / 18
        # first_line_angle = angles[0]
        # similar_lines = [hull[0][0].tolist()]  # 包含第一个点

        # # 找出方向近似相等的线段
        # for i in range(1, len(angles)):
        #     if abs(first_line_angle - angles[i]) < angle_threshold:
        #         similar_lines.append(hull[i][0].tolist())
        # for line in similar_lines:
        #     print(line)
        # cv2.line(img, line[0], line[1], (0, 0, 255), 2)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)

    # hexagons = find_hexagon(img, contours)
    # sorted_hexagons = sorted(hexagons, key=lambda hexagon: hexagon.lenght, reverse=True)

    # img1 = img.copy()
    # approx1 = cv2.approxPolyDP(cnt, 3, True)  # 拟合精确度
    # img1 = cv2.polylines(img1, [approx1], True, (255, 255, 0), 2)
    # cv2.imshow("approxPolyDP1", img1)
    # cv2.waitKey(0)
    # img2 = img.copy()
    # approx2 = cv2.approxPolyDP(cnt, 5, True)  # 拟合精确度
    # img2 = cv2.polylines(img2, [approx2], True, (255, 255, 0), 2)
    # cv2.imshow("approxPolyDP2", img2)
    # cv2.waitKey(0)
    # img3 = img.copy()
    # approx3 = cv2.approxPolyDP(cnt, 10, True)  # 拟合精确度
    # img3 = cv2.polylines(img3, [approx3], True, (255, 255, 0), 5)
    # cv2.imshow("approxPolyDP3", img3)
    # hs = hexagons[0]
    # h1 = hexagons[0]
    # he = hexagons[0]
    # for h in hexagons:
    #     print(h.start, h.end, h.lenght)
    #     if h1.start == h.start:
    #         hs = h
    #         h1 = h
    #         continue
    #     if h1.end == h.start:
    #         he = h
    #     if h1.end != h.start:
    #         print("not found", h1.start, h1.end, h.start, h.end)
    #         cv.line(img, hs.start, h1.end, (0, 0, 225), 2)
    #         cv.line(img, hs.start, he.end, (0, 0, 225), 2)
    #         # cv.imshow("result", img)
    #         # cv.waitKey(0)
    #         hs = h
    #     h1 = h

    # for i in range(6):
    #     # sorted_hexagons[0]和sorted_hexagons[1]的交点
    #     s = i
    #     e = (i + 1) % 6
    #     line1 = [sorted_hexagons[s].start, sorted_hexagons[s].end]
    #     line2 = [sorted_hexagons[e].start, sorted_hexagons[e].end]
    #     cv.line(img, sorted_hexagons[s].start, sorted_hexagons[s].end, (0, 0, 225), 2)
    #     x, y = intersection(line1, line2)
    #     cv.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
    #     cv.imshow("result", img)
    #     cv.waitKey(0)
    # 显示
    cv.imshow("result", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def detect_cricle():
    src = cv.imread("img/2.jpg")
    img = cv.resize(src, (0, 0), fx=0.3, fy=0.3)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    cv.waitKey(0)
    # 获取结构元素
    k = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # 开操作
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, k)
    cv.imshow("result", binary)
    cv.waitKey(0)
    # 进行中值滤波
    # dst_img = cv.medianBlur(img_gray, 7)

    # 霍夫圆检测
    circles = cv.HoughCircles(
        binary,
        cv.HOUGH_GRADIENT,
        1,
        10,
        param1=10,
        param2=100,
        minRadius=0,
        maxRadius=10000,
    )

    print(circles)
    arr1 = np.zeros([0, 2], dtype=int)  # 创建一个0行, 2列的空数组
    if circles is not None:
        circles = np.uint16(np.around(circles))  # 4舍5入, 然后转为uint16
        for i in circles[0, :]:
            arr1 = np.append(arr1, (i[0], i[1]))  # arr1是圆心坐标的np数组
            # print(arr1)
            cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 255), 3)  # 轮廓
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 0), 6)  # 圆心
    # # 将检测结果绘制在图像上
    # for i in circle[0, :]:  # 遍历矩阵的每一行的数据
    #     # 绘制圆形
    #     cv.circle(img, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 10)
    #     # 绘制圆心
    #     cv.circle(img, (int(i[0]), int(i[1])), 10, (255, 0, 0), -1)

    # 显示图像
    cv.imshow("result", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    # detect_cricle()
    detect_hull()

    # img = cv2.imread("img/4.jpg")
    # img = cv2.resize(img, (0, 0), fx=0.45, fy=0.45)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 100, 150, apertureSize=3)

    # minLineLength = 12
    # maxLineGap = 8
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 5, minLineLength, maxLineGap, 25)
    # for line in lines:
    #     for x1, y1, x2, y2 in line:
    #         cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # cv2.imshow("image", edges)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    # 点与轮廓关系

    # 求解图像中的一个点到一个对象轮廓的最短距离。如果点在轮廓的外部，返回值为负。如果在轮廓上，返回值为 0。如果在轮廓内部，返回值为正。

    # 下面我们以点（50，50）为例：
    # dist = cv2.pointPolygonTest(cnt,(50,50),True)
    # 此函数的第三个参数是 measureDist。如果设置为 True，就会计算最短距离。如果是 False，只会判断这个点与轮廓之间的位置关系（返回值为+1，-1，0）。

    # 形状匹配

    # 函数 cv2.matchShape() 可以帮我们比较两个形状或轮廓的相似度。如果返回值越小，匹配越好。它是根据 Hu 矩来计算的。文档中对不同的方法都有解释。

    # import cv2
    # import numpy as np
    # img1 = cv2.imread('star.jpg',0)
    # img2 = cv2.imread('star2.jpg',0)
    # ret, thresh = cv2.threshold(img1, 127, 255,0)
    # ret, thresh2 = cv2.threshold(img2, 127, 255,0)
    # contours,hierarchy = cv2.findContours(thresh,2,1)
    # cnt1 = contours[0]
    # contours,hierarchy = cv2.findContours(thresh2,2,1)
    # cnt2 = contours[0]
    # ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
    # print ret
