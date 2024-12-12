import open3d as o3d
import numpy as np
import os

def display_reconstruction(data, data_type):
    # important note: may be point cloud or mesh input
    if data_type == 'mesh':
        data.paint_uniform_color([0.8, 0.8, 0.8])
        data.compute_vertex_normals()
            
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(data)
    vis.run()
    vis.destroy_window()
def process_point_cloud(pcd, output_dir):
    # display initial point cloud
    display_reconstruction(pcd, 'pc')
    # Remove outlier points
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=60, std_ratio=1.5)
    pcd = pcd.select_by_index(ind)
    o3d.io.write_point_cloud(f"{output_dir}remove_outlier_points.ply", pcd)
    display_reconstruction(pcd, 'pc')
    # Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size=0.02)
    o3d.io.write_point_cloud(f"{output_dir}downsample.ply", pcd_down)
    display_reconstruction(pcd_down, 'pc')
    # Estimate normals
    pcd_down.estimate_normals()
    pcd_down.orient_normals_consistent_tangent_plane(100)
    # Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_down, depth=7)
    o3d.io.write_triangle_mesh(f"{output_dir}poisson_reconstruction.ply", mesh)
    display_reconstruction(mesh, 'mesh')
    # Density based filtering
    remove_faces = []
    for i, face in enumerate(mesh.triangles):
        face_density = np.mean([densities[vertex] for vertex in face])
        if face_density < 4:
            remove_faces.append(i)
    mesh.remove_triangles_by_index(remove_faces)
    o3d.io.write_triangle_mesh(f"{output_dir}minimum_density_threshold.ply", mesh)
    display_reconstruction(mesh, 'mesh')
    # Remove mesh outliers + smooth mesh
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.filter_smooth_simple(number_of_iterations=5)
    #mesh.paint_uniform_color([0.8, 0.8, 0.8])
    #mesh.compute_vertex_normals()
    display_reconstruction(mesh, 'mesh')
    return mesh

def main():
    output_dir = 'pcd_to_mash_output_steps/'
    filename = "enhanced_SuiteScan.ply"
    pcd = o3d.io.read_point_cloud(filename)
    mesh = process_point_cloud(pcd, output_dir)
    print("saving mesh to ply")
    o3d.io.write_triangle_mesh(f"{output_dir}processed_{filename}", mesh)
    print("done saving mesh")
        


if __name__ == '__main__':
    main()