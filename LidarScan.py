import open3d as o3d
import numpy as np

# Load point cloud
pcd = o3d.io.read_point_cloud("SuiteScan.ply")

# Simple visualization
o3d.visualization.draw_geometries([pcd])



# Customize point cloud visualization
pcd.paint_uniform_color([0, 0.7, 0])  # Color the point cloud green
o3d.visualization.draw_geometries([pcd])



# Create mesh from point cloud
pcd.estimate_normals()
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# Clean the mesh
mesh.remove_duplicate_vertices()
mesh.remove_degenerate_triangles()
mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Gray color

# Visualize reconstructed mesh
o3d.visualization.draw_geometries([mesh])



# Alternative mesh reconstruction
pcd.estimate_normals()
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radii = [avg_dist, avg_dist*2]
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

# Visualize
o3d.visualization.draw_geometries([mesh])



# Detailed rendering with multiple processing steps
def process_point_cloud(pcd):
    # Downsample to reduce computational load
    pcd_down = pcd.voxel_down_sample(voxel_size=0.05)
    
    # Estimate normals for better reconstruction
    pcd_down.estimate_normals()
    pcd_down.orient_normals_consistent_tangent_plane(100)
    
    # Poisson reconstruction
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_down, depth=9)
    
    # Optional: remove outliers and smooth the mesh
    mesh.remove_duplicate_vertices()
    mesh.remove_degenerate_triangles()
    mesh.filter_smooth_simple(number_of_iterations=1)
    
    # Add some color and rendering properties
    mesh.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray
    mesh.compute_vertex_normals()
    
    return mesh

# Process and render
mesh = process_point_cloud(pcd)
o3d.visualization.draw_geometries([mesh])

# Render with custom rendering options
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(mesh)

# Customize rendering
render_option = vis.get_render_option()
render_option.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background
render_option.point_size = 1.0
render_option.line_width = 1.0

vis.run()
vis.destroy_window()


