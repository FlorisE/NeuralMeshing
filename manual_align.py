import argparse
import copy
import json
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Align two meshes")
    parser.add_argument("--source", type=str, required=True, help="Path to source mesh")
    parser.add_argument("--target", type=str, required=True, help="Path to target mesh")
    parser.add_argument(
        "--transforms_in",
        help="Path to the source transforms file",
        required=True,
        nargs='+', default=[]
    )
    parser.add_argument(
        "--transforms_out",
        help="Path to the target transforms file",
        required=True,
        nargs='+', default=[]
    )
    parser.add_argument(
        "--kp_in",
        type=str,
        help="Path to a keypoints file to automatically apply",
        required=False,
    )
    parser.add_argument(
        "--kp_out",
        type=str,
        help="Path to optionally write keypoints to",
        required=False,
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        help="Scale factor for the meshes, which will be undone",
        required=False,
        default=1.0,
    )
    return parser.parse_args()


def pick_points(mesh):
    print("Pick at least three correspondences using [shift + left click]")
    print("Press [shift + right click] to undo point picking")
    print("After picking points, press [Q] to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()
    return vis.get_picked_points()


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def register(source, target, picked_id_source, picked_id_target, threshold):
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(
        source, target, o3d.utility.Vector2iVector(corr)
    )
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    draw_registration_result(source, target, reg_p2p.transformation)
    return reg_p2p.transformation


def align(align_target, align_subject, transforms_in, transforms_out, scale, kp_out, kp_in):
    if isinstance(align_target, Path):
        align_target = str(align_target)
    if isinstance(align_subject, Path):
        align_subject = str(align_subject)
    if isinstance(transforms_in, Path):
        transforms_in = str(transforms_in)
    if isinstance(transforms_out, Path):
        transforms_out = str(transforms_out)
    if isinstance(kp_out, Path):
        kp_out = str(kp_out)
    if isinstance(kp_in, Path):
        kp_in = str(kp_in)

    initial_source = align_subject
    initial_target = align_target
    transformation = None
    with open(transforms_in) as f:
        transforms = json.load(f)

    transforms_np = [
        np.array(frame["transform_matrix"]) for frame in transforms["frames"]
    ]

    if transformation is None:
        source_mesh = o3d.io.read_triangle_mesh(initial_source)
        target_mesh = o3d.io.read_triangle_mesh(initial_target)
        if scale != 1.0:
            source_mesh.scale(1 / scale, center=source_mesh.get_center())
            target_mesh.scale(1 / scale, center=target_mesh.get_center())
        source = o3d.geometry.PointCloud()
        source.points = source_mesh.vertices
        source.colors = source_mesh.vertex_colors
        target = o3d.geometry.PointCloud()
        target.points = target_mesh.vertices
        target.colors = target_mesh.vertex_colors
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(source.points)
        extent = bbox.extent
        diag = np.linalg.norm(extent)
        if not kp_in:
            while True:
                print("Select the same number of points on both meshes")
                picked_id_source = pick_points(source)
                picked_id_target = pick_points(target)
                if len(picked_id_source) >= 3 and len(picked_id_target) == len(
                    picked_id_source
                ):
                    if kp_out:
                        json_out = {"source": picked_id_source, "target": picked_id_target}
                        with open(kp_out, "w") as f:
                            json.dump(json_out, f, indent=2)

                    break
        else:
            kp_in = json.load(open(kp_in))
            picked_id_source = kp_in["source"]
            picked_id_target = kp_in["target"]
        transformation = register(
            source, target, picked_id_source, picked_id_target, diag / 200.0
        )
    T1 = np.eye(4)
    T1[:3, :3] = Rotation.from_rotvec([-np.pi / 2, 0, 0]).as_matrix()
    T2 = np.eye(4)
    T2[:3, :3] = Rotation.from_rotvec([0, -np.pi / 2, 0]).as_matrix()
    T3 = np.eye(4)
    T3[:3, :3] = Rotation.from_rotvec([0, np.pi / 2, 0]).as_matrix()
    T4 = np.eye(4)
    T4[:3, :3] = Rotation.from_rotvec([np.pi / 2, 0, 0]).as_matrix()
    for i, transform in enumerate(transforms_np):
        transformed = T1 @ transform
        transformed = T2 @ transformed
        transformed = np.matmul(transformation, transformed)
        transformed = T3 @ transformed
        transformed = T4 @ transformed
        transforms_np[i] = transformed

    transforms_copy = copy.deepcopy(transforms)

    for i, frame in enumerate(transforms_copy["frames"]):
        frame["transform_matrix"] = transforms_np[i].tolist()

    with open(transforms_out, "w") as f:
        json.dump(transforms_copy, f, indent=2)



if __name__ == "__main__":
    args = parse_args()

    align(args.source, args.target, args.transforms_in, args.scale_factor, args.kp_out, args.kp_in, args.transforms_out)