import os
import sys
import numpy as np
import open3d as o3d

def read_txt(txt_path):
  txt_path = './img/13.txt'
  # 通过numpy读取txt点云
  pcd = np.genfromtxt(txt_path, delimiter=",")
  
  pcd_vector = o3d.geometry.PointCloud()
  # 加载点坐标
  pcd_vector.points = o3d.utility.Vector3dVector(pcd[:, :3])
  o3d.visualization.draw_geometries([pcd_vector])
  # o3d.io.write_point_cloud("./img/13.pcd", pcd_vector)
  o3d.io.write_point_cloud("./img/13.ply", pcd_vector)

def read_pcd(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    o3d.visualization.draw_geometries([pcd])


#open3d安装：pip3 install open3d-python
import open3d as o3d
import numpy as np

def makePlyFile(xyzs, labels, fileName='makeply.ply'):
    '''Make a ply file for open3d.visualization.draw_geometries
    :param xyzs:    numpy array of point clouds 3D coordinate, shape (numpoints, 3).
    :param labels:  numpy array of point label, shape (numpoints, ).
    '''
    RGBS = [
        (0, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (0, 0, 0),
        (255, 0, 255)
    ]

    with open(fileName, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('comment PCL generated\n')
        f.write('element vertex {}\n'.format(len(xyzs)))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for i in range(len(xyzs)):
            r, g, b = (255, 0, 0)

            x, y, z = xyzs[i]
            f.write('{} {} {} {} {} {}\n'.format(x, y, z, r, g, b))

def txt2ply(fileName):
    xyzs=[]
    with open(fileName, "r") as f:
        for line in f.readlines():
            p=[]
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            a=line.split(",")
            p.append(float(a[0]))
            p.append(float(a[1]))
            p.append(float(a[2]))
            xyzs.append(p)

    numpoints=len(xyzs)
    labels = np.random.randint(0, 4, numpoints)  # 4种label
    makePlyFile(xyzs, labels, 'demo.ply')
    pcd = o3d.io.read_point_cloud('demo.ply')
    o3d.visualization.draw_geometries([pcd])

def ply2pcd(filename):
    mesh_ply = o3d.geometry.TriangleMesh()
 
    # load ply
    mesh_ply = o3d.io.read_triangle_mesh(filename)
    mesh_ply.compute_vertex_normals()
 
    # V_mesh 为ply网格的顶点坐标序列，shape=(n,3)，这里n为此网格的顶点总数，其实就是浮点型的x,y,z三个浮点值组成的三维坐标
    V_mesh = np.asarray(mesh_ply.vertices)
    # F_mesh 为ply网格的面片序列，shape=(m,3)，这里m为此网格的三角面片总数，其实就是对顶点序号（下标）的一种组合，三个顶点组成一个三角形
    F_mesh = np.asarray(mesh_ply.triangles)
 
    print("ply info:", mesh_ply)
    print("ply vertices shape:", V_mesh.shape)
    print("ply triangles shape:", F_mesh.shape)
    o3d.visualization.draw_geometries([mesh_ply], window_name="ply", mesh_show_wireframe=True)
 
    # # ply -> stl
    # mesh_stl = o3d.geometry.TriangleMesh()
    # mesh_stl.vertices = o3d.utility.Vector3dVector(V_mesh)
    # mesh_stl.triangles = o3d.utility.Vector3iVector(F_mesh)
    # mesh_stl.compute_vertex_normals()
    # print("stl info:",mesh_stl)
    # o3d.visualization.draw_geometries([mesh_stl], window_name="stl")
    # o3d.io.write_triangle_mesh(os.path.join(root_folder, "test_data", "Bunny.stl"), mesh_stl)
 
    # stl/ply -> pcd
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(V_mesh)
    print("pcd info:",pcd)
    o3d.visualization.draw_geometries([pcd],window_name="pcd")
 
    # save pcd
    o3d.io.write_point_cloud("./img/133.pcd",pcd)

def obj2pcd(filename):
    objfile = filename
    points, _ = read_obj(objfile)
    pcdfile = objfile.replace('.obj', '.pcd')
    save_pcd(pcdfile, points)
 
def read_obj(obj_path):
    with open(obj_path) as file:
        points = []
        faces = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "f":
                faces.append((int(strs[1]), int(strs[2]), int(strs[3])))
    points = np.array(points)
    faces = np.array(faces)
 
    return points, faces
 
 
def save_pcd(filename, pcd):
    num_points = np.shape(pcd)[0]
    f = open(filename, 'w')
    f.write('# .PCD v0.7 - Point Cloud Data file format \nVERSION 0.7 \nFIELDS x y z \nSIZE 4 4 4 \nTYPE F F F \nCOUNT 1 1 1 \n')
    f.write('WIDTH {} \nHEIGHT 1 \nVIEWPOINT 0 0 0 1 0 0 0 \n'.format(num_points))
    f.write('POINTS {} \nDATA ascii\n'.format(num_points))
    for i in range(num_points):
        new_line = str(pcd[i,0]) + ' ' + str(pcd[i,1]) + ' ' + str(pcd[i,2]) + '\n'
        f.write(new_line)
    f.close()

if __name__ == "__main__":
#   obj2pcd("./img/BunnyMesh.ply")
    print("hello")
