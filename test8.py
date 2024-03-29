import open3d as o3d
 
if __name__ == "__main__":
    # mesh = o3d.io.read_triangle_mesh("./img/13.ply")
    # o3d.visualization.draw_geometries([mesh])
 
    # print("Displaying pointcloud with convex hull ...")
    bunny = o3d.data.BunnyMesh()
    mesh:o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh("./img/13.ply")
    # mesh:o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(bunny.path)
    mesh.paint_uniform_color([0,0,1])#给mesh上色，颜色为蓝色
    mesh.compute_vertex_normals()#计算mesh中点的法向量
 
    #因为样例中加载的是Stanford bunny的mesh格式，
    # 因此此处通过sample_points_poisson_disk，
    # 将mesh采样生成点云，并通过点云计算convex hull
    # pcl = mesh.sample_points_poisson_disk(number_of_points=16200000)
    pcl = mesh.sample_points_uniformly(number_of_points=10000)
    hull, _ = pcl.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))
    o3d.visualization.draw_geometries([pcl, hull_ls])

  # print("Displaying pointcloud with convex hull ...")
  # mesh = o3d.io.read_triangle_mesh("./img/13.ply")
  # o3d.visualization.draw_geometries([mesh])

  # mesh.compute_vertex_normals()
  # #网格点采样
  # pcl = mesh.sample_points_poisson_disk(number_of_points=2000)
  # # # 计算点云凸包
  # hull, _ = pcl.compute_convex_hull()
  # # # 计算网格凸包
  # # #hull, _ = mesh.compute_convex_hull()
  # # #创建线集
  # hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
  # hull_ls.paint_uniform_color((1, 0, 0))
  # o3d.visualization.draw([pcl, hull_ls])
