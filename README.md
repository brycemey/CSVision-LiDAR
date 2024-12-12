### 3D Reconstruction of College Dorm Suite

SuiteScan.ply - initial LiDAR scanning data of test scene

Learning3D - this repository gives us point cloud deep learning capabilities (PCN) and is required for predictive completion enhancement

Run the following command to test the PCN model on the test scene point cloud:
```
python pcn_enhancement.py
```
This will generate the `enhanced_SuiteScan.ply` point cloud file

Next, run 
```
python pcn_to_mesh.py
```
to run the enhanced point cloud through further point cloud processing, mesh rendering, and mesh processing. This file will send step-by-step reconstruction (point cloud or mesh .ply files) outputs to the `pcd_to_mash_output_steps/` folder 

This `pcn_to_mesh.py` file performs the following procedures on the enhanced point cloud:
1. Statistical outlier removal from enhanced point cloud
2. Voxel downsampling
3. Poisson reconstruction (this step turns the point cloud representations into mesh renderings)
4. Minimum density thresholding
5. Duplicated vertex & degenerate triangle removal
6. Average filtering
