import open3d as o3d
import numpy as np
import copy


import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN


def fit_circle(points):
    # 使用最小二乘法拟合圆
    # 参考: https://www.cnblogs.com/zyly/p/9405102.html
    points = np.array(points)
    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    b = points[:, 2]
    # 解线性方程组 Ax = b
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    # 圆心和半径
    center = np.array([x[0], x[1]])
    radius = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2])
    return center, radius


def detect_circle_holes(point_cloud, distance_threshold=0.1, min_samples=10):
    # 转换为 NumPy 数组
    points = np.asarray(point_cloud.points)
    # 使用 DBSCAN 聚类算法来查找可能属于圆孔的点
    clustering = DBSCAN(eps=distance_threshold, min_samples=min_samples).fit(
        points[:, :2]
    )
    labels = clustering.labels_

    # 对于每个聚类，尝试拟合一个圆
    circles = []
    for label in set(labels):
        if label == -1:
            # -1 表示噪声点，跳过
            continue
        cluster_points = points[labels == label]
        center, radius = fit_circle(cluster_points)
        circles.append((center, radius, cluster_points))

    return circles


# 加载点云或创建一个示例点云
# 这里我们创建一个包含圆孔的简单点云
def create_circle_point_cloud(radius, height, num_points=100):
    points = []
    for theta in np.linspace(0, 2 * np.pi, num_points):
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = height * np.random.rand() - height / 2
        points.append([x, y, z])
    return np.array(points)


def create_circle_mesh(radius, resolution=30):
    # 创建一个圆形的三角网格
    points = []
    triangles = []
    for i in range(resolution):
        theta = 2 * np.pi * i / resolution
        points.append([radius * np.cos(theta), radius * np.sin(theta), 0])
    points.append([0, 0, 0])  # 圆心
    for i in range(resolution):
        triangles.append([i, (i + 1) % resolution, resolution])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh


# 创建一个包含圆孔的点云
points = np.genfromtxt("img/13.txt", delimiter=",")
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)


# # 检测圆孔
# circles = detect_circle_holes(point_cloud)

# # 可视化结果
# for center, radius, cluster_points in circles:
#     print(f"Circle detected at center {center} with radius {radius}")
#     # 绘制圆
#     circle_mesh = create_circle_mesh(radius)
#     circle_mesh.translate(center)
#     o3d.visualization.draw_geometries([point_cloud, circle_mesh])

# 显示点云
o3d.visualization.draw_geometries([point_cloud])
