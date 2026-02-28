
import os
import sys
import numpy as np
import open3d as o3d
from utils import read_canonical_model, load_pc, visualize_icp_result

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
  sys.path.insert(0, CODE_DIR)

from icp.icp_3d_warmup import icp_3d_multi_init, transform_points_3d
from icp.correspondences import nearest_neighbors


def compute_rmse(source_pc, target_pc, pose, max_dist=0.2):
  source_t = transform_points_3d(source_pc, pose)
  src_idx, dst_idx, dist = nearest_neighbors(source_t, target_pc, max_dist=max_dist)
  if dist.size == 0:
    return None
  return float(np.sqrt(np.mean(dist ** 2)))


def save_icp_result_png(source_pc, target_pc, pose, out_path, width=1280, height=720):
  source_pcd = o3d.geometry.PointCloud()
  source_pcd.points = o3d.utility.Vector3dVector(source_pc.reshape(-1, 3))
  source_pcd.paint_uniform_color([0, 0, 1])

  target_pcd = o3d.geometry.PointCloud()
  target_pcd.points = o3d.utility.Vector3dVector(target_pc.reshape(-1, 3))
  target_pcd.paint_uniform_color([1, 0, 0])

  source_pcd.transform(pose)

  vis = o3d.visualization.Visualizer()
  vis.create_window(width=width, height=height, visible=False)
  vis.add_geometry(source_pcd)
  vis.add_geometry(target_pcd)
  vis.poll_events()
  vis.update_renderer()
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  vis.capture_screen_image(out_path, do_render=True)
  vis.destroy_window()


if __name__ == "__main__":
  obj_name = 'drill' # drill or liq_container
  num_pc = 4 # number of point clouds
  save_png = True

  source_pc = read_canonical_model(obj_name)

  for i in range(num_pc):
    target_pc = load_pc(obj_name, i)

    # estimated_pose from ICP
    pose, hist = icp_3d_multi_init(source_pc, target_pc, num_yaw=24)
    rmse = compute_rmse(source_pc, target_pc, pose)
    print(f"{obj_name} pc{i} rmse: {rmse}, mse_last: {hist[-1] if hist else None}")

    # visualize the estimated result
    visualize_icp_result(source_pc, target_pc, pose)
    if save_png:
      out_path = os.path.join("code", "icp_warm_up", "output", f"{obj_name}_pc{i}.png")
      save_icp_result_png(source_pc, target_pc, pose, out_path)
