## 3D Reconstruction of College Dorm Suite

### Background File Info
SuiteScan.ply - initial LiDAR scanning data of test scene
https://github.com/brycemey/CSVision-LiDAR/blob/westondcrewe/README.md
Open3D - the library we used for point cloud visualization and mesh rendering

Learning3D - this repository gives us point cloud deep learning capabilities (PCN) and is required for predictive completion enhancement

### Setup
We used a virtual environment to run our repository on Python3.9, allowing us to install the right versions of library / package dependencies in their required versions
Before running any of our files, it is essential to spin up the [Learning3D](https://github.com/vinits5/learning3d/tree/master) repository, which contains the Point Completion Network (PCN) deep learning model and pretrained model weights (model pretrained on the [ModelNet40](https://www.kaggle.com/datasets/balraj98/modelnet40-princeton-3d-object-dataset) dataset)

### Run Instructions
If you wish, you can first run the PCN model on the testing split of the provided ModelNet40 dataset:
```
python learning3d/examples/test_pcn.py 
```
Loss is evaulated by Chamfer Distance calculation between input and predicted point clouds.
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

### Example Results
Initial suite scan \
<img width="599" alt="suite_scan_2" src="https://github.com/user-attachments/assets/aeab266e-aba1-4b4e-89c2-876f2c4f4f7a" /> \
Outlier removal \
<img width="617" alt="suite_outlierremoval" src="https://github.com/user-attachments/assets/fc749004-869b-4ef0-81b0-d578f6d890ed" /> \
Voxel downsampling \
<img width="595" alt="suite_voxeldownsampling" src="https://github.com/user-attachments/assets/b8558219-2f9c-4830-8b22-d6852a56a9d5" /> \
Poisson reconstruction \
<img width="733" alt="suite_poissonreconstruction" src="https://github.com/user-attachments/assets/2bce4c62-d7c0-4fc5-9324-92c85cded7d5" /> \
Minimum vertex density treshold \
<img width="580" alt="suite_minvertexdensitythreshold" src="https://github.com/user-attachments/assets/a255199e-6ba9-4371-b647-9bf884d8b31f" /> \
Final mesh \
<img width="623" alt="suite_mesh_2" src="https://github.com/user-attachments/assets/057b76f7-fb48-4b40-a9b4-93191c26f860" /> \
