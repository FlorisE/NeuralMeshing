"""
NeuralMeshing pipeline
"""

import argparse
from bop_toolkit_lib import inout
from collections import defaultdict
import cv2
import json
import numpy as np
import open3d as o3d
import os
from pathlib import Path
import pycolmap
import shutil
import sys
from tqdm import tqdm
import yaml
import logging
from manual_align import align

SELECT_ROI_WINDOW = 'Select ROI'

logger = logging.getLogger('NeuralMeshing')
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Runs the whole NeuralMeshing pipeline")
    parser.add_argument("input_folder", type=Path, help="Path to the input folder")
    parser.add_argument("--object_id", type=int, help="BOP ID of the object", default=0)
    parser.add_argument("--config", type=Path, help="Path to the config file", required=False, default="config.yaml")
    parser.add_argument("--environment", type=str, help="Environment to process", required=False, default=None)
    return parser.parse_args()


def project_pixel_to_3d_ray(uv, cameraMatrix):
    x = (uv[0,0] - cameraMatrix[0, 2]) / cameraMatrix[0, 0]
    y = (uv[0,1] - cameraMatrix[1, 2]) / cameraMatrix[1, 1]
    # norm = math.sqrt(x**2 + y**2 + 1)
    # x /= norm
    # y /= norm
    # z = 1.0 / norm
    z = 1.0
    return np.array([x, y, z])


def get_scene_camera(transforms):
    scene_camera = {}

    cam_K = [transforms["fl_x"], 0.0, transforms["cx"]]
    cam_K += [0.0, transforms["fl_y"], transforms["cy"]]
    cam_K += [0.0, 0.0, 1.0]

    depth_scale = 1.0

    for i, frame in enumerate(transforms["frames"]):
        index = Path(frame["file_path"]).stem

        scene_camera[str(int(index))] = {
            "cam_K": cam_K,
            "depth_scale": depth_scale,
            "resolution": [transforms["w"], transforms["h"]],
            "height": transforms["h"],
            "width": transforms["w"],
        }

    return scene_camera


def copy_images(source_dir, target_dir, scene):
    os.makedirs(target_dir, exist_ok=True)
    for f in os.listdir(source_dir):
        shutil.copy(os.path.join(source_dir, f), os.path.join(target_dir, f))


rotation_180 = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
rotate_neg_90_y = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
rotate_90_z = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def get_scene_gt(transforms, world_to_model, object_id):
    model_to_world = np.linalg.inv(world_to_model)
    canonical_model_to_colmap_model = model_to_world
    s = np.linalg.norm(canonical_model_to_colmap_model[:3, 0])
    canonical_model_to_colmap_model_rescaled = canonical_model_to_colmap_model.copy()
    canonical_model_to_colmap_model_rescaled[:3, :3] /= s

    scene_gt = {}
    for frame in transforms["frames"]:
        frame_idx = int(Path(frame["file_path"]).stem)
        camera_to_world = np.array(frame["transform_matrix"])
        world_to_camera = np.linalg.inv(camera_to_world)

        model_to_camera = rotation_180 @ world_to_camera @ rotate_neg_90_y @ rotate_90_z
        m2c = model_to_camera @ canonical_model_to_colmap_model_rescaled
        gt = {}
        gt["cam_R_m2c"] = m2c[:3, :3].flatten().tolist()
        gt["cam_t_m2c"] = (m2c[:3, -1] / s).tolist()
        gt["obj_id"] = object_id
        scene_gt[str(int(frame_idx))] = [gt]

    return scene_gt


def run_ffmpeg(images, video_in, video_fps):
    logger.info(f"Running ffmpeg on {images} and {video_in} with {video_fps} fps.")
    # from instant-ngp's colmap2nerf
    ffmpeg_binary = "ffmpeg"

    if os.name == "nt" and os.system(f"where {ffmpeg_binary} >nul 2>nul") != 0:
        ffmpeg_glob = os.path.join(ROOT_DIR, "external", "ffmpeg", "*", "bin", "ffmpeg.exe")
        candidates = glob(ffmpeg_glob)
        if not candidates:
            logger.info("FFmpeg not found. Attempting to download FFmpeg from the internet.")
            os.system(os.path.join(SCRIPTS_FOLDER, "download_ffmpeg.bat"))
            candidates = glob(ffmpeg_glob)

        if candidates:
            ffmpeg_binary = candidates[0]

    if not os.path.isabs(images):
        images = os.path.join(os.path.dirname(video_in), images)

    images = "\"" + images + "\""
    video =  "\"" + video_in + "\""
    fps = float(video_fps) or 1.0
    logger.info(f"running ffmpeg with input video file={video}, output image folder={images}, fps={fps}.")
    try:
        shutil.rmtree(images)
    except:
        pass
    os.system(f"mkdir {images}")
    os.system(f"{ffmpeg_binary} -i {video} -qscale:v 1 -qmin 1 -vf \"fps={fps}\" {images}/%04d.jpg")


class NeuralMeshing:
    def __init__(self, args, config):
        self.instant_ngp_path = os.getenv("INSTANT_NGP_PATH")
        self.neus2_export_mesh_path = os.getenv("NEUS2_EXPORT_MESH_PATH")
        self.neus2_original_export_mesh_path = os.getenv("NEUS2_ORIGINAL_EXPORT_MESH_PATH")
        self.ingp_export_mesh_path = os.getenv("INGP_EXPORT_MESH_PATH")
        self.input_folder = args.input_folder
        self.environments_folder = self.input_folder / "environments"
        self.object_id = args.object_id
        self.config = config
        self.environment = args.environment

    def verify_instant_ngp(self):
        if Path('dependencies/instant-ngp').is_dir():
            self.instant_ngp_path = Path('dependencies/instant-ngp').absolute()
        elif not self.instant_ngp_path:
            logger.error("INSTANT_NGP_PATH environment variable not set")
            exit()
        else:
            self.instant_ngp_path = Path(self.instant_ngp_path).absolute()
        if not self.instant_ngp_path.is_dir():
            logger.error("INSTANT_NGP_PATH is not a valid directory")
            exit()

    def get_environments(self, segmented=False):
        if not self.environment and not segmented:
            return [environment for environment in sorted(os.listdir(self.environments_folder)) if not environment.startswith('.') and not environment.startswith('_') and not environment.endswith('_segmented')]
        if not self.environment and segmented:
            return [environment for environment in sorted(os.listdir(self.environments_folder)) if not environment.startswith('.') and not environment.startswith('_') and environment.endswith('_segmented')]
        if self.environment and not segmented:
            return [self.environment]
        if self.environment and segmented:
            return [self.environment + "_segmented"]

    def run_glomap(self):
        self.verify_instant_ngp()
        video_folder = self.input_folder / "videos"
        if not os.path.isdir(video_folder):
            logger.error("Video folder does not exist")
            exit(1)

        work_dir = os.getcwd()

        for video_path in sorted(os.listdir(video_folder)):
            video_name = video_path.split(".")[0]
            extract_images = not (self.environments_folder / video_name / 'images').exists()
            videos_folder = self.input_folder / "videos"
            if extract_images:
                os.makedirs(self.environments_folder, exist_ok=True)
                if Path(videos_folder / video_name).is_dir():
                    shutil.copytree(videos_folder / video_name, self.environments_folder / video_name / 'images')
                else:
                    if video_path.endswith('mp4'):
                        os.makedirs(self.environments_folder / video_name, exist_ok=True)
                        shutil.copy(videos_folder / video_path, self.environments_folder / video_name / 'video.mp4')
                    else:
                        logger.info(f"Unsupported video format: {video_path}")
                        continue
            os.chdir(self.environments_folder / video_name)
            images_path = Path('images')
            if extract_images:
                video_fps = self.config['video']['video_fps']
                run_ffmpeg(str(images_path.absolute()), 'video.mp4', video_fps)
            os.system(f"colmap feature_extractor --image_path {images_path} --database_path colmap.db --ImageReader.camera_model OPENCV --ImageReader.single_camera 1 --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true")
            os.system(f"colmap exhaustive_matcher --database_path colmap.db")
            os.system(f"glomap mapper --image_path {images_path} --database_path colmap.db --output_path colmap_sparse")
            os.system(f"colmap bundle_adjuster --input_path colmap_sparse/0 --output_path colmap_sparse/0 --BundleAdjustment.refine_principal_point 0")
            os.makedirs('colmap_text', exist_ok=True)
            os.system(f"colmap model_converter --input_path colmap_sparse/0 --output_path colmap_text --output_type TXT")
            os.system(f"python {self.instant_ngp_path / 'scripts' / 'colmap2nerf.py'}")
            os.chdir(work_dir)

    def run_colmap(self):
        self.verify_instant_ngp()
        video_folder = self.input_folder / "videos"
        if not os.path.isdir(video_folder):
            logger.error("Video folder does not exist")
            exit(1)

        work_dir = os.getcwd()

        for video_path in sorted(os.listdir(video_folder)):
            video_name = video_path.split(".")[0]
            extract_images = not (self.environments_folder / video_name / 'images').exists()
            if extract_images:
                os.makedirs(self.environments_folder, exist_ok=True)
                if video_path.endswith('mp4'):
                    os.makedirs(self.environments_folder / video_name, exist_ok=True)
                    shutil.copy(self.input_folder / "videos" / video_path, self.environments_folder / video_name / 'video.mp4')
                else:
                    logger.info(f"Unsupported video format: {video_path}")
                    continue
            os.chdir(self.environments_folder / video_name)
            images_path = Path('images')
            if extract_images:
                video_fps = self.config['video']['video_fps']
                os.system(f"python {self.instant_ngp_path / 'scripts' / 'colmap2nerf.py'} --run_colmap --video_in video.mp4 --video_fps {video_fps} --colmap_camera_model OPENCV --overwrite --colmap_matcher exhaustive")
            else:
                os.system(f"python {self.instant_ngp_path / 'scripts' / 'colmap2nerf.py'} --run_colmap --images {images_path.absolute()} --colmap_camera_model OPENCV --overwrite --colmap_matcher exhaustive")
            num_colmap_input_images = len(os.listdir(images_path))
            num_frames = len(json.load(open('transforms.json'))['frames'])
            if num_frames < num_colmap_input_images * 0.8:
                logger.error(f'Number of frames ({num_frames}) is less than 80% of the number of COLMAP input images ({num_colmap_input_images})')
                exit()
            os.chdir(work_dir)

    def select_calibration_frames(self):
        logger.info('Select frames to use for calibration')
        logger.info('Controls:')
        logger.info('A: Add frame to calibration set')
        logger.info('S: Skip frame')
        logger.info('N: Move to next scene')
        logger.info('Q: Quit')
        environment_frames = defaultdict(list)
        for environment in self.get_environments():
            logger.info(f"Environment: {environment}")
            index = 0
            frames = sorted(os.listdir(self.environments_folder / environment / "images"))
            while index < len(frames):
                frame = frames[index]
                image = cv2.imread(str(self.environments_folder / environment / "images" / frame))
                cv2.imshow('Frame', image)
                key = cv2.waitKey(0)
                if key == ord('a'):
                    logger.info('Adding frame')
                    environment_frames[environment].append(frame)
                elif key == ord('s'):
                    logger.info('Skipping frame')
                elif key == ord('n'):
                    logger.info('Moving to next environment')
                    break
                elif key == ord('q'):
                    logger.info('Quitting')
                    exit()
                else:
                    logger.info('Invalid key')
                    index -= 1
                index += 1
            cv2.destroyWindow('Frame')
        with open(f"{self.input_folder / 'calibration_frames.json'}", 'w') as f:
            json.dump(environment_frames, f)

    def calibrate(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        rows = self.config["checkerboard"]["rows"]
        cols = self.config["checkerboard"]["cols"]
        objp = np.zeros(((rows-1)*(cols-1),3), np.float32)
        objp[:,:2] = np.mgrid[0:cols-1,0:rows-1].T.reshape(-1,2)
        objp *= self.config["checkerboard"]["square_size"]
        objpoints = []
        imgpoints = []
        calibration_frames = json.load(open(self.input_folder / "calibration_frames.json"))
        for environment in tqdm(self.get_environments()):
            objpoints_env = []
            imgpoints_env = []
            for frame in calibration_frames[environment]:
                image = cv2.imread(str(self.environments_folder / environment / "images" / frame))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (cols-1, rows-1), None)
                if ret:
                    objpoints_env.append(objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                    imgpoints_env.append(corners2)
                else:
                    logger.info(f"Could not find checkerboard corners in frame {frame}")
            objpoints.append(objpoints_env)
            imgpoints.append(imgpoints_env)
        # Flatten objpoints and imgpoints
        objpoints = [item for sublist in objpoints for item in sublist]
        imgpoints = [item for sublist in imgpoints for item in sublist]
        environment_ids = []
        for environment in self.get_environments():
            # Insert environment id len(calibration_frames) times
            for calibration_frame in calibration_frames[environment]:
                environment_ids.append((environment, calibration_frame))
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        with open(self.input_folder / "distances.json", 'w') as f:
            distances = defaultdict(dict)
            for i in range(len(tvecs)):
                distances[environment_ids[i][0]][environment_ids[i][1]] = {
                    'x': round(imgpoints[i][0][0][0]),
                    'y': round(imgpoints[i][0][0][1]),
                    'd': np.linalg.norm(tvecs[i]),
                    'mtx': mtx.tolist(),
                    'distCoeffs': dist.tolist()
                }
            json.dump(distances, f, indent=2)

    def write_calibration_frames(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        rows = self.config["checkerboard"]["rows"]
        cols = self.config["checkerboard"]["cols"]
        objp = np.zeros(((rows-1)*(cols-1),3), np.float32)
        objp[:,:2] = np.mgrid[0:cols-1,0:rows-1].T.reshape(-1,2)
        objp *= self.config["checkerboard"]["square_size"]
        objpoints = []
        imgpoints = []
        calibration_frames = json.load(open(self.input_folder / "calibration_frames.json"))
        for environment in tqdm(self.get_environments()):
            objpoints_env = []
            imgpoints_env = []
            for frame in calibration_frames[environment]:
                image = cv2.imread(str(self.environments_folder / environment / "images" / frame))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (cols-1, rows-1), None)
                if ret:
                    objpoints_env.append(objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                    imgpoints_env.append(corners2)
                else:
                    logger.info(f"Could not find checkerboard corners in frame {frame}")
            objpoints.append(objpoints_env)
            imgpoints.append(imgpoints_env)
        # Flatten objpoints and imgpoints
        objpoints = [item for sublist in objpoints for item in sublist]
        imgpoints = [item for sublist in imgpoints for item in sublist]
        environment_ids = []
        for environment in self.get_environments():
            # Insert environment id len(calibration_frames) times
            for calibration_frame in calibration_frames[environment]:
                environment_ids.append((environment, calibration_frame))
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        count = 0
        for e_i, environment in enumerate(self.get_environments()):
            os.makedirs(self.environments_folder / environment / "calibration_frames", exist_ok=True)
            for frame in calibration_frames[environment]:
                image = cv2.imread(str(self.environments_folder / environment / "images" / frame))
                cv2.drawFrameAxes(image, mtx, dist, rvecs[count], tvecs[count], 0.1)
                cv2.imwrite(f'{self.environments_folder / environment / "calibration_frames" / frame}', image)
                count += 1

    def train_nerfs(self, segmented):
        if not segmented:
            self.verify_instant_ngp()
            snapshot_suffix = "base.ingp"
            transforms_suffix = "transforms.json"
        else:
            self.verify_instant_ngp()
            snapshot_suffix = "base_segmented.ingp"
            transforms_suffix = "transforms_segmented.json"
        for environment in self.get_environments():
            env_path = (self.environments_folder / environment).absolute()
            snapshot_str = str(env_path / snapshot_suffix)
            transforms_str = str(env_path / transforms_suffix)
            os.system(f"python {self.instant_ngp_path / 'scripts' / 'run.py'} --scene {transforms_str} --n_steps {self.config['nerf_training']['steps']} --save_snapshot {snapshot_str}")

    def estimate_scale(self):
        self.verify_instant_ngp()
        sys.path.append(str(self.instant_ngp_path / "build"))
        import pyngp as ngp  # noqa

        with open(self.input_folder / "distances.json") as f:
            distances = json.load(f)
        sys.path.append(str(self.instant_ngp_path / "build"))

        environment_scales = {}
        for environment in self.get_environments():
            testbed = ngp.Testbed()
            testbed.load_snapshot(str((self.environments_folder / environment / "base.ingp").absolute()))
            testbed.background_color = [0.0, 0.0, 0.0, 0.0]
            testbed.render_mode = ngp.RenderMode.Depth
            testbed.nerf.render_gbuffer_hard_edges = True
            testbed.shall_train = False

            frames = []
            with open(self.environments_folder / environment / "transforms.json") as f:
                transforms = json.load(f)
                for frame in transforms['frames']:
                    file_path = Path(frame['file_path'])
                    frames.append(file_path.name)

            distances_for_environment = {}
            for key, value in distances[environment].items():
                distances_for_environment[key] = value

            sorted_frames = sorted(frames)
            scales = []
            for idx in tqdm(range(testbed.nerf.training.dataset.n_images)):
                testbed.set_camera_to_training_view(idx)
                resolution = testbed.nerf.training.dataset.metadata[idx].resolution
                output = testbed.render(*resolution, 1, True)
                alpha = output[:, :, -1:]
                depth_image = output[:, :, :1]
                depth_image = np.clip(depth_image, 0.0, 10.0)
                depth_image = depth_image[:, :, 0]
                alpha = alpha[:, :, 0]
                depth_image = cv2.medianBlur(depth_image, 5)
                # fn_stem = Path(testbed.nerf.training.dataset.paths[idx]).stem
                # depth_output_fn = (depth_dir / fn_stem).with_suffix(".png")
                # inout.save_depth(str(depth_output_fn), depth_image * 1000)
                if sorted_frames[idx] in distances_for_environment.keys():
                    marker_properties = distances_for_environment[sorted_frames[idx]]
                    distance = marker_properties['d']
                    u = marker_properties['x']
                    v = marker_properties['y']
                    mtx = np.array(marker_properties['mtx'])
                    distCoeffs = np.array(marker_properties['distCoeffs'])
                    uv = np.array([[float(u), float(v)]])
                    rect = cv2.undistortPoints(uv, mtx, distCoeffs)
                    ray = project_pixel_to_3d_ray(rect[0], mtx)
                    ray *= depth_image[int(v), int(u)]
                    d_scale = distance / np.linalg.norm(ray)
                    scales.append(d_scale)
            environment_scales[environment] = {'mean': np.mean(scales),
                                               'scales': scales}
        with open(self.input_folder / "scales.json", 'w') as f:
            json.dump(environment_scales, f, indent=2)

    def scale_nerfs_to_uniform(self):
        environment_scales = json.load(open(self.input_folder / "scales.json"))

        scale_factors = {}
        for i, environment in enumerate(self.get_environments()):
            scale = environment_scales[environment]['mean']
            if i == 0:
                target_scale = scale
                scale_factors[environment] = 1.0
                shutil.copy(self.environments_folder / environment / "transforms.json", self.environments_folder / environment / "transforms_uniform_scale.json")
                continue
            scale_factor = scale / target_scale
            logger.info(f"Scale factor for {environment}: {scale_factor}")
            scale_factors[environment] = scale_factor
            environment_folder = self.environments_folder / environment 
            with open(environment_folder / "transforms.json", 'r') as json_file:
                transforms = json.load(json_file)
                # transforms['fl_x'] *= scale_factor
                # transforms['fl_y'] *= scale_factor
                for frame in transforms['frames']:
                    for x in range(3):
                        frame['transform_matrix'][x][3] *= scale_factor
            with open(environment_folder / "transforms_uniform_scale.json", 'w') as json_file:
                json.dump(transforms, json_file, indent=2)

        with open(self.input_folder / "scale_factors.json", 'w') as f:                                                                                    
            json.dump(scale_factors, f, indent=2)

    def retrain_rescaled_nerfs(self):
        self.verify_instant_ngp()
        for environment in self.get_environments():
            env_path = (self.environments_folder / environment).absolute()
            snapshot_str = str(env_path / "base_uniform_scale.ingp")
            transforms_str = str(env_path / "transforms_uniform_scale.json")
            os.system(f"python {self.instant_ngp_path / 'scripts' / 'run.py'} --scene {transforms_str} --n_steps {self.config['nerf_training']['steps']} --save_snapshot {snapshot_str}")

    def rois(self):
        boxes = {}
        for environment in self.get_environments():
            environment_folder = self.environments_folder / environment
            frame_names = sorted(os.listdir(self.input_folder / "environments" / environment / "images"))
            img_path = frame_names[0]
            img = cv2.imread(environment_folder / "images" / img_path)
            while True:
                x, y, w, h = cv2.selectROI(SELECT_ROI_WINDOW, img, showCrosshair=False)
                cv2.destroyWindow(SELECT_ROI_WINDOW)
                if x == y == w == h == 0:
                    logger.info('Empty ROI')
                    continue
                break
            boxes[environment] = {'x': x, 'y': y, 'w': w, 'h': h}
        with open(self.input_folder / "rois.json", 'w') as f:
            json.dump(boxes, f, indent=2)

    def segment(self):
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        device = torch.device("cuda")
        boxes = json.load(open(self.input_folder / "rois.json"))
        for environment in self.get_environments():
            environment_folder = self.environments_folder / environment
            segmented_folder = self.environments_folder / (environment + "_segmented")
            frame_names = sorted(os.listdir(self.input_folder / "environments" / environment / "images"))
            box_json = boxes[environment]
            box = np.array([box_json['x'], box_json['y'], box_json['x'] + box_json['w'], box_json['y'] + box_json['h']])
            predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml", self.config['segmentation']['sam2_checkpoint'], device=device)
            inference_state = predictor.init_state(video_path=str(environment_folder / "images"))
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                box=box
            )
            os.makedirs(environment_folder / "segmentation", exist_ok=True)
            os.makedirs(environment_folder / "images_png", exist_ok=True)
            os.makedirs(segmented_folder / "images", exist_ok=True)
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                out_mask = out_mask_logits[0] > 0.0
                mask_np = out_mask.cpu().numpy()[0] * 255
                cv2.imwrite(str(environment_folder / "segmentation" / f"{out_frame_idx+1:04d}.png"), mask_np)
                frame_img = cv2.imread(environment_folder / "images" / frame_names[out_frame_idx])
                image_copy = cv2.cvtColor(frame_img, cv2.COLOR_BGR2BGRA)
                image_copy[:, :, 3] = mask_np
                cv2.imwrite(str(environment_folder / "images_png" / f"{out_frame_idx+1:04d}.png"), image_copy)
            with open(environment_folder / "transforms_segmented_uniform_scale.json", 'w') as f:
                with open(environment_folder / "transforms_uniform_scale.json") as f2:
                    transforms = json.load(f2).copy()
                    transforms['aabb_scale'] = 1.0
                    for frame in transforms['frames']:
                        file_path = Path(frame['file_path'])
                        frame['file_path'] = f"images_png/{file_path.stem + '.png'}"
                    json.dump(transforms, f, indent=2)
            with open(environment_folder / "transforms_segmented.json", 'w') as f:
                with open(environment_folder / "transforms.json") as f2:
                    transforms = json.load(f2).copy()
                    transforms['aabb_scale'] = 1.0
                    for frame in transforms['frames']:
                        file_path = Path(frame['file_path'])
                        frame['file_path'] = f"images_png/{file_path.stem + '.png'}"
                    json.dump(transforms, f, indent=2)

    def generate_segmented_features(self):
        for environment in self.get_environments():
            environment_folder = self.environments_folder / environment
            # copy segmented images to new folder
            segmented_folder = self.environments_folder / (environment + "_segmented")
            os.makedirs(segmented_folder / "images", exist_ok=True)
            to_skip = []
            for f in sorted(os.listdir(environment_folder / "images_png")):
                img = cv2.imread(environment_folder / "images_png" / f, cv2.IMREAD_UNCHANGED)
                mask = img[..., 3] == 0
                img[mask, :3] = 0
                gray = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
                non_zero_count = cv2.countNonZero(gray)
                if non_zero_count == 0:
                    logger.info(f"Empty frame: {f}")
                    to_skip.append(f)
                    continue
                else:
                    cv2.imwrite(segmented_folder / "images" / f, img)
            shutil.copy(environment_folder / "transforms_segmented.json", segmented_folder / "transforms.json")
            shutil.copytree(environment_folder / "colmap_text", segmented_folder / "colmap_text", dirs_exist_ok=True)
            with open(segmented_folder / "colmap_text" / "images.txt", "r+") as f:
                contents = f.readlines()
                for i in range(4, len(contents), 2):
                    contents[i] = contents[i].replace('jpg', 'png')
                for i in range(5, len(contents)+1, 2):
                    contents[i] = "\n"
                f.seek(0)
                f.truncate(0)
                for i in range(4):
                    f.write(contents[i])
                idx = 1
                for i in range(4, len(contents), 2):
                    skip_this = False
                    parts = contents[i].split(" ")
                    original_id = parts[-1]
                    for skip in to_skip:
                        if original_id.startswith(skip):
                            skip_this = True
                            break
                    if not skip_this:
                        f.write(contents[i])
                        f.write(contents[i+1])
            db_path = segmented_folder / "colmap_text" / "database.db"
            img_dir = segmented_folder / "images"
            pycolmap.extract_features(db_path, img_dir, camera_mode=pycolmap.CameraMode.SINGLE, extraction_options={'sift': {'estimate_affine_shape': True, 'domain_size_pooling': True}})
            matching_options = pycolmap.FeatureMatchingOptions()
            matching_options.guided_matching = True
            pycolmap.match_exhaustive(db_path, matching_options=matching_options)
            # reconstruction = pycolmap.Reconstruction(segmented_folder / "colmap_text")

            # triangulated_reconstruction = pycolmap.triangulate_points(reconstruction=reconstruction,
            #                                                           database_path=db_path,
            #                                                           image_path=img_dir,
            #                                                           output_path=segmented_folder / "colmap_text")
            # reconstruction.write_text(segmented_folder / "colmap_text")

    def merge_segmented_features(self):
        environment_points = {}
        for environment in self.get_environments(segmented=True):
            environment_folder = self.environments_folder / environment
            points = []
            colors = []
            with open(environment_folder / "colmap_text" / "points3D.txt", "r") as file:
                for line in file.readlines():
                    if len(line.strip()) and line.strip()[0] == '#':
                        continue
                    point = np.array(list(map(float, line.split(' ')[1:4])))
                    color = np.array(list(map(float, line.split(' ')[4:7])))
                    point[0], point[1] = point[1], point[0]
                    point[2] *= -1
                    points.append(point)
                    colors.append(color)
            # environment_points[environment] = np.array(points)
            points = np.array(points)
            colors = np.array(colors)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
            o3d.io.write_point_cloud(str(environment_folder / "points3D.ply"), pcd)

    def estimate_depth(self):
        self.verify_instant_ngp()
        sys.path.append(str(self.instant_ngp_path / "build"))
        import pyngp as ngp  # noqa

        for environment in self.get_environments():
            testbed = ngp.Testbed()
            testbed.load_snapshot(str((self.environments_folder / environment / "base_uniform_scale.ingp").absolute()))
            testbed.background_color = [0.0, 0.0, 0.0, 0.0]
            testbed.render_mode = ngp.RenderMode.Depth
            testbed.nerf.render_gbuffer_hard_edges = True
            testbed.shall_train = False

            os.makedirs(self.environments_folder / environment / "depth", exist_ok=True)

            depth_dir = self.environments_folder / environment / "depth"
            if not depth_dir.exists():
                depth_dir.mkdir()
            for idx in tqdm(range(testbed.nerf.training.dataset.n_images)):
                testbed.set_camera_to_training_view(idx)
                resolution = testbed.nerf.training.dataset.metadata[idx].resolution
                output = testbed.render(*resolution, 1, True)
                alpha = output[:, :, -1:]
                depth_image = output[:, :, :1]
                depth_image = depth_image[:, :, 0]
                alpha = alpha[:, :, 0]
                depth_image = cv2.medianBlur(depth_image, 5)
                fn_stem = Path(testbed.nerf.training.dataset.paths[idx]).stem
                depth_output_fn = (depth_dir / fn_stem).with_suffix(".png")
                inout.save_depth(str(depth_output_fn), depth_image * 1000)

    def generate_gt_bop(self):
        for environment in self.get_environments():
            environment_folder = self.environments_folder / environment
            bop_folder = environment_folder / "bop_raw"
            with open(environment_folder / "transforms_uniform_scale.json") as f:
                transforms = json.load(f)
            os.makedirs(bop_folder, exist_ok=True)
            scene_camera = get_scene_camera(transforms)
            scene_camera_fn = bop_folder / "scene_camera.json"
            scene_camera_fn = scene_camera_fn.with_stem(scene_camera_fn.stem + "_initial")
            with open(scene_camera_fn, "w") as fp:
                json.dump(scene_camera, fp, indent=2)
            copy_images(environment_folder / "images", bop_folder / "rgb", environment)
            copy_images(environment_folder / "depth", bop_folder / "depth_nerf", environment)
            copy_images(environment_folder / "segmentation", bop_folder / "segmentation", environment)
            alignment_transform_mtx = np.eye(4)
            scene_gt = get_scene_gt(transforms, alignment_transform_mtx, self.object_id)
            scene_gt_fn = bop_folder / "scene_gt_initial.json"
            with open(scene_gt_fn, "w") as fp:
                json.dump(scene_gt, fp, indent=2)

    def generate_tsdf_meshes(self):
        import open3d as o3d

        def apply_mask(image, mask):
            image = np.asarray(image)
            mask = np.asarray(mask)
            image[mask == 0] = 0
            image = o3d.geometry.Image(image)
            return image

        environments = self.get_environments()
        pbar = tqdm(environments, position=0, total=len(environments))
        for environment in pbar:
            pbar.description = f'Processing {environment}'
            environment_folder = self.environments_folder / environment
            bop_folder = environment_folder / 'bop_raw'

            with open(bop_folder / 'scene_gt_initial.json') as f:
                bop_poses = json.load(f)
            with open(bop_folder / 'scene_camera_initial.json') as f:
                bop_cameras = json.load(f)

            bop_poses = dict(sorted(bop_poses.items(), key=lambda key: int(key[0])))
            bop_cameras = dict(sorted(bop_cameras.items(), key=lambda key: int(key[0])))

            camera_trajectory_list = []
            for pose, camera in zip(bop_poses.values(), bop_cameras.values()):
                pose = pose[0]
                extrinsics = np.eye(4)
                extrinsics[:3, :3] = np.array(pose['cam_R_m2c']).reshape(3, 3)
                extrinsics[:3, 3] = np.array(pose['cam_t_m2c'])

                K = np.array(camera['cam_K']).reshape(3, 3)
                camera_params = o3d.camera.PinholeCameraParameters()
                camera_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(width=int(camera['width']),
                                                                            height=int(camera['height']),
                                                                            fx=K[0, 0],
                                                                            fy=K[1, 1],
                                                                            cx=K[0, 2],
                                                                            cy=K[1, 2])
                camera_params.extrinsic = extrinsics
                camera_trajectory_list.append(camera_params)

            camera_trajectory = o3d.camera.PinholeCameraTrajectory()
            camera_trajectory.parameters = camera_trajectory_list

            rgbd_images = []
            rgb_files = sorted((bop_folder / 'rgb').glob('*.jpg'))
            depth_files = sorted((bop_folder / 'depth_nerf').glob('*.png'))
            mask_files = sorted((bop_folder / 'segmentation').glob('*.png'))

            for i in range(len(rgb_files)):
                rgb_file = rgb_files[i]
                depth_file = depth_files[i]
                mask_file = mask_files[i]
                rgb_image = o3d.io.read_image(str(rgb_file))
                depth_image = o3d.io.read_image(str(depth_file))
                mask_image = o3d.io.read_image(str(mask_file))

                rgb_image = apply_mask(rgb_image, mask_image)
                depth_image = apply_mask(depth_image, mask_image)

                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image, depth_scale=1000.0, depth_trunc=1e6, convert_rgb_to_intensity=False)
                rgbd_images.append(rgbd_image)

            volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=4.0 / 512,
                                                                  sdf_trunc=0.04,
                                                                  color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

            for rgbd_image, camera_params in tqdm(zip(rgbd_images, camera_trajectory.parameters)):
                volume.integrate(rgbd_image, camera_params.intrinsic, camera_params.extrinsic)

            pcd = volume.extract_point_cloud()
            pcd, _ = pcd.remove_radius_outlier(nb_points=500, radius=0.05 * 5)
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=500, std_ratio=1.0)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.05, max_nn=30
                )
            )
            pcd.orient_normals_consistent_tangent_plane(50)
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=7)
            vertices_to_remove = densities < np.quantile(densities, 0.01)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            mesh = mesh.filter_smooth_taubin(number_of_iterations=10, lambda_filter=0.2, mu=-0.21)
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()

            max_repair_attempts = 3
            for _ in range(max_repair_attempts):
                if mesh.is_edge_manifold() and mesh.is_vertex_manifold():
                    break
                else:
                    logger.info("\tMesh is not manifold, attempting to repair ...")
                cluster_idx_per_triangle, num_triangles_per_cluster, _ = map(np.asarray, mesh.cluster_connected_triangles())
                triangles_to_remove = num_triangles_per_cluster[cluster_idx_per_triangle] < num_triangles_per_cluster.max()
                mesh.remove_triangles_by_mask(triangles_to_remove)
                mesh.remove_non_manifold_edges()
            if not mesh.is_edge_manifold() or not mesh.is_vertex_manifold():
                logger.info("\tMesh is not manifold after repair attempts, skipping ...")

            mesh = mesh.subdivide_loop(number_of_iterations=1)
            mesh, camera_trajectory = o3d.pipelines.color_map.run_non_rigid_optimizer(mesh,
                                                                                      rgbd_images,
                                                                                      camera_trajectory,
                                                                                      o3d.pipelines.color_map.NonRigidOptimizerOption(
                                                                                          maximum_iteration=50,
                                                                                          maximum_allowable_depth=10.0))
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            o3d.t.io.write_triangle_mesh(str(environment_folder / 'mesh.obj'), mesh)
            mesh.compute_vertex_normals()
            mesh.compute_uvatlas()
            mesh.vertex['diffuse'] = mesh.vertex.colors
            mesh.material.set_default_properties()
            texture_tensors = mesh.bake_vertex_attr_textures(1024, {'diffuse'})
            texture_diffuse = texture_tensors['diffuse'].numpy()
            texture_diffuse = (texture_diffuse * 255).astype(np.uint8)
            texture_diffuse = cv2.cvtColor(texture_diffuse, cv2.COLOR_RGB2BGR)
            cv2.imwrite(environment_folder / 'texture_diffuse.png', texture_diffuse)
            del mesh.triangle.normals
            del mesh.vertex.colors
            o3d.t.io.write_triangle_mesh(str(environment_folder / 'mesh_textured.obj'), mesh)
            with open(environment_folder / 'mesh_textured.mtl', 'a') as f:
                f.write('map_Kd texture_diffuse.png\n')

    def generate_neus2_meshes(self):
        environments = self.get_environments()

        for environment in environments:
            logger.info(f"Generating NeuS2 mesh for environment: {environment}")
            environment_folder = (Path(self.environments_folder) / environment).resolve()
            os.system(f"{self.neus2_export_mesh_path} {environment_folder}/transforms_segmented_uniform_scale.json {environment_folder}/neus2_mesh.obj")

    def align_meshes(self):
        environments = self.get_environments()

        if len(environments) == 1:
            logger.info('Only one environment found, no alignment needed')
            return  # nothing to align
        
        env_base = environments[0]
        env_base_folder = self.environments_folder / env_base

        align_target_path = Path(env_base_folder) / 'neus2_mesh.obj'

        if not align_target_path.exists():
            logger.error(f'Base environment mesh not found: {align_target_path}')
            exit(1)

        for i in range(1, len(environments)):
            env_subject = environments[i]
            env_subject_folder = self.environments_folder / env_subject

            # use the updated mesh as target except for the first mesh
            subject_path = Path(env_subject_folder) / 'neus2_mesh.obj'

            subject_transforms_path = Path(env_subject_folder) / 'transforms_segmented_uniform_scale.json'
            out_transforms_path = Path(env_subject_folder) / 'transforms_aligned_to_previous_env.json'
            
            if not align_target_path.exists() or \
               not subject_transforms_path.exists():
                logger.error('Invalid inputs')
                exit(1)

            kp_out_path = env_subject_folder / 'keypoints_aligned_to_previous_env.json'
            if kp_out_path.exists():
                align(align_target_path, subject_path, subject_transforms_path, out_transforms_path, 1.0, kp_out_path, kp_out_path)
            else:
                align(align_target_path, subject_path, subject_transforms_path, out_transforms_path, 1.0, kp_out_path, None)

    def merge_aligned(self):
        environments = self.get_environments()

        if len(environments) == 1:
            logger.info('Only one environment found, no merging needed')
            return
        
        merged_dir = self.input_folder / 'merged'
        merged_dir.mkdir(exist_ok=True)

        transforms_merged = json.load(open(self.environments_folder / environments[0] / 'transforms_segmented_uniform_scale.json'))
        frames = transforms_merged['frames']
        for frame in frames:
            frame['file_path'] = frame['file_path'].replace('images_png/', f'{environments[0]}_images/')

        for i, environment in enumerate(environments):
            environment_image_folder = merged_dir / f'{environment}_images'
            if environment_image_folder.exists():
                shutil.rmtree(environment_image_folder)
            shutil.copytree(self.environments_folder / environment / 'images_png', environment_image_folder)
            if i != 0:
                environment_transforms = json.load(open(self.environments_folder / environment / 'transforms_aligned_to_previous_env.json'))
                frames = environment_transforms['frames']
                for frame in frames:
                    frame['file_path'] = frame['file_path'].replace('images_png/', f'{environment}_images/')
                transforms_merged['frames'].extend(frames)
        
        json.dump(transforms_merged, open(merged_dir / 'transforms.json', 'w'), indent=2)

    def generate_merged_mesh(self):
        mesh_folder = (Path(self.input_folder) / 'merged').resolve()
        os.system(f"{self.neus2_export_mesh_path} {mesh_folder}/transforms.json {mesh_folder}/mesh_uniform_scale.obj")

    def generate_merged_mesh_neus2_original(self):
        mesh_folder = (Path(self.input_folder) / 'merged').resolve()
        os.system(f"{self.neus2_original_export_mesh_path} {mesh_folder}/transforms.json {mesh_folder}/mesh_uniform_scale_neus2_original.obj")

    def generate_merged_mesh_ingp(self):
        mesh_folder = (Path(self.input_folder) / 'merged').resolve()
        os.system(f"{self.ingp_export_mesh_path} {mesh_folder}/transforms.json {mesh_folder}/mesh_uniform_scale_ingp.obj")

    def scale_mesh(self):
        mesh_folder = Path(self.input_folder) / 'merged'
        environment0 = self.get_environments()[0]
        environment_scales = json.load(open(self.input_folder / "scales.json"))
        scale = environment_scales[environment0]['mean']
        mesh = o3d.io.read_triangle_mesh(f"{mesh_folder}/mesh_uniform_scale.obj")
        mesh.scale(scale, center=mesh.get_center())
        o3d.io.write_triangle_mesh(f"{mesh_folder}/mesh.obj", mesh)

def neural_meshing_factory():
    args = parse_args()
    logger.info("Arguments:")
    logger.info(f"Input folder: {args.input_folder}")
    config = yaml.safe_load(open(args.config))
    logger.info("Config:")
    for key, value in config.items():
        logger.info(f"{key}: {value}")

    neural_meshing = NeuralMeshing(args, config)
    return neural_meshing, config


if __name__ == "__main__":
    neural_meshing, config = neural_meshing_factory()

    logger.info("Running NeuralMeshing pipeline")

    def scale_nerf():
        neural_meshing.scale_nerfs_to_uniform()
        neural_meshing.retrain_rescaled_nerfs()

    steps = {
        "skip_glomap": lambda: neural_meshing.run_glomap(),
        # "skip_colmap": lambda: neural_meshing.run_colmap(),
        "skip_select_calibration_frames": lambda: neural_meshing.select_calibration_frames(),
        "skip_calibrate": lambda: neural_meshing.calibrate(),
        "skip_write_calibration_frames": lambda: neural_meshing.write_calibration_frames(),
        "skip_train_unsegmented_nerfs": lambda: neural_meshing.train_nerfs(segmented=False),
        "skip_estimate_scale": lambda: neural_meshing.estimate_scale(),
        "skip_scale_nerf": lambda: scale_nerf(),
        "skip_rois": lambda: neural_meshing.rois(),
        "skip_segment": lambda: neural_meshing.segment(),
        "skip_generate_segmented_features": lambda: neural_meshing.generate_segmented_features(),
        "skip_merge_segmented_features": lambda: neural_meshing.merge_segmented_features(),
        "skip_estimate_depth": lambda: neural_meshing.estimate_depth(),
        "skip_train_segmented_nerfs": lambda: neural_meshing.train_nerfs(segmented=True),
        "skip_generate_gt_bop": lambda: neural_meshing.generate_gt_bop(),
        "skip_generate_tsdf_mesh": lambda: neural_meshing.generate_tsdf_meshes(),
        "skip_generate_neus2_mesh": lambda: neural_meshing.generate_neus2_meshes(),
        "skip_align": lambda: neural_meshing.align_meshes(),
        "skip_merge_aligned": lambda: neural_meshing.merge_aligned(),
        "skip_generate_merged_mesh": lambda: neural_meshing.generate_merged_mesh(),
        "skip_generate_merged_mesh_neus2_original": lambda: neural_meshing.generate_merged_mesh_neus2_original(),
        "skip_generate_merged_mesh_ingp": lambda: neural_meshing.generate_merged_mesh_ingp(),
        "skip_scale_mesh": lambda: neural_meshing.scale_mesh()
    }

    for key, step in steps.items():
        if config[key]:
            logger.info(f"=== Skipping {key[5:]} ===")
        else:
            logger.info(f"=== Running {key[5:]} ===")
            step()

    logger.info("Done")
