import open3d as o3d
import numpy as np
import os
 
 
def pca_compute(data, sort=True):
    """
    点云的特征值与特征向量
    :return: 特征值与特征向量
    """
    average_data = np.mean(data, axis=0)
    decentration_matrix = data - average_data
 
    H = np.dot(decentration_matrix.T, decentration_matrix) 
    eigenvectors, eigenvalues, eigenvectors_T = np.linalg.svd(H) 
    if sort:
        sort = eigenvalues.argsort()[::-1] 
        eigenvalues = eigenvalues[sort] 
    return eigenvalues
 
def caculate_surface_curvature(radius,pcd):
    """
    计算点云的表面曲率
    :return: 点云中每个点的表面曲率
    """
    cloud = pcd
    points = np.asarray(cloud.points)
    kdtree = o3d.geometry.KDTreeFlann(cloud)
    num_points = len(cloud.points)
 
    curvature = []  # 储存表面曲率
    for i in range(num_points):
        k, idx, _ = kdtree.search_radius_vector_3d(cloud.points[i], radius)
 
        neighbors = points[idx, :]
        w = pca_compute(neighbors)  
        delt = np.divide(w[2], np.sum(w), out=np.zeros_like(w[2]), where=np.sum(w) != 0)
        curvature.append(delt)
    curvature = np.array(curvature, dtype=np.float64)
 
    return curvature
 
def curvature_normal():
    '''
    传入的曲率curvature归一化，传出诡异数据到绘图模块
    :return:
    '''
    data_normal = caculate_surface_curvature(radius,pcd)
    ave = np.mean(data_normal)
    data_max = max(data_normal)
    data_min = min(data_normal)
    cur_normal = [(float(i) - data_min) / (data_max - data_min) for i in data_normal]
 
    return cur_normal
 
def draw(cur_max,cur_min,pcd):
    '''
    绘图法向量绘图，曲率可视化绘图
    :return:
    '''
    cur_normal = curvature_normal()
    downpcd_normals = pcd
    print(pcd)
    print(cur_normal)
 
    pcd.paint_uniform_color([0.5,0.5,0.5])
    for i in range(len(cur_normal)):
        if 0 < cur_normal[i] <= cur_min:
            np.asarray(pcd.colors)[i] = [1, 0, 0]
        elif cur_min < cur_normal[i] <= cur_max:
            np.asarray(pcd.colors)[i] = [0, 1, 0]
        elif cur_max < cur_normal[i] <= 1:
            np.asarray(pcd.colors)[i] = [0, 0, 1]
 
    # 曲率分割基准
    o3d.visualization.draw_geometries([downpcd_normals],window_name="可视化原始点云",
                                      width=800, height=800, left=50, top=50,
                                      mesh_show_back_face=False)
    return None
 
def save_txt1(cur_min,filename):
    '''
    存1列txt
    :return:
    '''
    un1 = []
    cur_normal = curvature_normal()
    for i in range(len(cur_normal)):
        if cur_normal[i] > cur_min:
            un1.append(i)
 
 
    savefilename = "%s"%(filename) + ".txt"
    savefilename = "./img/txt/" + savefilename
    if not os.path.exists(savefilename):
        f = open(savefilename, 'w')
        f.close()
    with open(savefilename, 'w') as file_to_write:
        for i in range(len(un1)):
            file_to_write.writelines(str(un1[i]) + "\n")
    return None
 
 
if __name__ == '__main__':
    cur_max = 0.7
    cur_min = 0.4
    radius = 0.07
    pcd = o3d.io.read_point_cloud("./img/13.txt") 
    caculate_surface_curvature(radius,pcd)
    print(caculate_surface_curvature(radius,pcd))
    curvature_normal()
    draw(cur_max,cur_min,pcd)
 
 
    filename = "data_cur"
    # 使用此函数记得更改文件名！！！！！！！！
    save_txt1(cur_min,filename)