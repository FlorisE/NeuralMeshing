# NeuralMeshing

Generate high quality meshes from casual captures of objects.
A fiducial marker is used for determining scene scale.

## Installation (Ubuntu)

Install everything into a single Python environment (e.g. venv):
```bash
python3 -m venv --prompt NeuralMeshing .venv
source .venv/bin/activate
```

Install _instant-ngp_ and _Segment Anything 2_ using the following commands:

For instant-ngp (https://github.com/NVlabs/instant-ngp), follow these instructions, but customize the CUDA version in lines 3 and 4:

```bash
sudo apt-get install build-essential git python3-dev python3-pip libopenexr-dev libxi-dev \
                     libglfw3-dev libglew-dev libomp-dev libxinerama-dev libxcursor-dev
export PATH="/usr/local/cuda-11.4/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH"
cd dependencies
git clone --recursive https://github.com/nvlabs/instant-ngp
cd instant-ngp
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j
pip install -r requirements.txt
cd ..
```

For Segment Anything 2:
```bash
cd dependencies
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2 & pip install -e .
```

Download a checkpoint from https://github.com/facebookresearch/segment-anything-2?tab=readme-ov-file#download-checkpoints (by default we use `sam2_hiera_large.pt` and place it in the `dependencies/segment-anything-2/checkpoints` folder.

Copy `config.yaml.sample` and modify it to your needs, save as `config.yaml`. The default values should suffice for running the full pipeline.

You need to set the following environment variables:
```bash
export INSTANT_NGP_PATH=/path/to/instant-ngp
export NEUS2_EXPORT_MESH_PATH=/path/to/our_neus2/export_mesh.sh
```

Use our NeuS2 fork from https://github.com/FlorisE/NeuS2.

Install COLMAP and GLOMAP as instructed here: https://github.com/colmap/glomap

For evaluation with `instant-ngp` and `NeuS2`, set the following environment variables:
```bash
export NEUS2_ORIGINAL_EXPORT_MESH_PATH=/path/to/original_neus2/export_mesh.sh
export INGP_EXPORT_MESH_PATH=/path/to/original/instant-ngp/export_mesh.sh
```

## Usage

Our pipeline is mostly automated, however a few manual steps using a GUI are required. Follow the instructions printed in the terminal and be ready to interact with the GUI.

1. Capture a scene with a checkerboard and a target object using a camera. Flip the object around to capture all sides. We need two or more videos.
2. Name your files `OBJECT_NAME_0.ext` (`mp4`, `mov`, etc.), `OBJECT_NAME_1.ext`, etc. and place them in a `OBJECT_NAME/videos` folder.
3. Start the pipeline by running `python main.py OBJECT_NAME` in the root directory of this repository.

To test out the pipeline, download sample data from https://drive.google.com/drive/folders/1BYjgqMutKnuD1zMLsWLogqdlV4tkLOBU?usp=sharing
