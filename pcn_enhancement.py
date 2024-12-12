# author: Vinit Sarode (vinitsarode5@gmail.com) 03/23/2020
# adapted by: Weston Crewe and Bryce Mey for CS153 Computer Vision

import open3d as o3d
import argparse
import os
import sys
import logging
import numpy
import numpy as np
import time
import torch
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from learning3d.models import PCN
from learning3d.data_utils import ModelNet40Data, ClassificationData
from learning3d.losses import ChamferDistanceLoss

def display_open3d(input_pc, output):
    input_pc_ = o3d.geometry.PointCloud()
    output_ = o3d.geometry.PointCloud()
    input_pc_.points = o3d.utility.Vector3dVector(input_pc)
    output_.points = o3d.utility.Vector3dVector(output + np.array([1, 0, 0]))  # Offset output for visual clarity
    input_pc_.paint_uniform_color([1, 0, 0])
    output_.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([input_pc_, output_])

def load_and_preprocess_point_cloud(file_path, num_points=1024):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return points

def run_pcn_on_point_cloud(model, device, input_points):
    input_points = torch.tensor(input_points, dtype=torch.float32).unsqueeze(0).to(device)
    output = model(input_points) 
    return output['coarse_output'].cpu().detach().numpy()[0]  

def test_one_epoch(device, model, test_loader, output_dir='output_point_clouds'):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	model.eval()
	test_loss = 0.0
	pred  = 0.0
	count = 0
	for i, data in enumerate(tqdm(test_loader)):
		points, _ = data
		points = points.to(device)

		output = model(points)
		loss_val = ChamferDistanceLoss()(points, output['coarse_output'])
		print("Loss Val: ", loss_val)

		predicted_points = output['coarse_output'].cpu().detach().numpy()[0]
		input_points = points.cpu().detach().numpy()[0]

		predicted_point_cloud = o3d.geometry.PointCloud()
		predicted_point_cloud.points = o3d.utility.Vector3dVector(predicted_points)
		o3d.io.write_point_cloud(os.path.join(output_dir, f'predicted_{i}.ply'), predicted_point_cloud)
		
		input_point_cloud = o3d.geometry.PointCloud()
		input_point_cloud.points = o3d.utility.Vector3dVector(input_points)
		o3d.io.write_point_cloud(os.path.join(output_dir, f'input_{i}.ply'), input_point_cloud)
		#display_open3d(points[0].detach().cpu().numpy(),output['coarse_output'][0].detach().cpu().numpy(),duration=0.2)  # Window stays open for 0.2 seconds before closing automatically)

		test_loss += loss_val.item()
		count += 1

	test_loss = float(test_loss)/count
	return test_loss

def options():
	parser = argparse.ArgumentParser(description='Point Completion Network')
	parser.add_argument('--exp_name', type=str, default='exp_pcn', metavar='N',
						help='Name of the experiment')
	parser.add_argument('--dataset_path', type=str, default='ModelNet40',
						metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
	parser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')

	# settings for input data
	parser.add_argument('--dataset_type', default='modelnet', choices=['modelnet', 'shapenet2'],
						metavar='DATASET', help='dataset type (default: modelnet)')
	parser.add_argument('--num_points', default=1024, type=int,
						metavar='N', help='points in point-cloud (default: 1024)')

	# settings for PCN
	parser.add_argument('--emb_dims', default=1024, type=int,
						metavar='K', help='dim. of the feature vector (default: 1024)')
	parser.add_argument('--detailed_output', default=False, type=bool,
						help='Coarse + Fine Output')

	# settings for on training
	parser.add_argument('--seed', type=int, default=1234)
	parser.add_argument('-j', '--workers', default=4, type=int,
						metavar='N', help='number of data loading workers (default: 4)')
	parser.add_argument('-b', '--batch_size', default=32, type=int,
						metavar='N', help='mini-batch size (default: 32)')
	parser.add_argument('--pretrained', default='learning3d/pretrained/exp_pcn/models/best_model.t7', type=str,
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')
	parser.add_argument('--device', default='cuda:0', type=str,
						metavar='DEVICE', help='use CUDA if available')

	args = parser.parse_args()
	return args

def main():
    args = options()
    point_cloud_file = 'SuiteScan.ply'
    # Load pretrained model
    model = PCN(emb_dims=args.emb_dims, detailed_output=args.detailed_output)
    args.pretrained = 'learning3d/pretrained/exp_pcn/models/best_model.t7'
    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        print("Loading pretrained model weights")
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
    #print("Saving model")
    #model.to(args.device)
    #model.eval()
    # 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load point cloud & run model
    input_points = load_and_preprocess_point_cloud(point_cloud_file)
    completed_points = run_pcn_on_point_cloud(model, device, input_points)

    # Visualize
    # red is input (scanned point cloud), green is output (predicted enhancement)
    display_open3d(input_points, completed_points)

    # Combine input and completed points to get enhanced point cloud
    combined_points = np.concatenate([input_points, completed_points], axis=0)
    combined_point_cloud = o3d.geometry.PointCloud()
    combined_point_cloud.points = o3d.utility.Vector3dVector(combined_points)
    output_file = f'enhanced_{point_cloud_file}'
    o3d.io.write_point_cloud(output_file, combined_point_cloud)

if __name__ == '__main__':
    main()
