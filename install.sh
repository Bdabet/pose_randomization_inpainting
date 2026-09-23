set -e
eval "$(conda shell.bash hook)"
# ######################## Phantom Env ###############################
# `conda create -n phantom` on an env that already exists silently WIPES and
# recreates it (no error, no confirmation) -- guard against destroying a
# previously-installed environment on a re-run.
if conda env list | grep -q "^phantom "; then
    echo "Conda env 'phantom' already exists -- reusing it (delete it first with 'conda env remove -n phantom' for a clean rebuild)."
else
    conda create -n phantom python=3.10 -y
fi
conda activate phantom
conda install nvidia/label/cuda-12.1.0::cuda-toolkit -c nvidia/label/cuda-12.1.0 -y

# Install SAM2
cd submodules/sam2
pip install -v -e ".[notebooks]"
cd ../..

# Install torch (needed before Hamer: its dependency detectron2 imports torch
# in its own setup.py at build time, but doesn't declare torch as a PEP 517
# build requirement, so it must already be installed in this environment).
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.1.0 torchvision==0.16.0
# setuptools>=81 dropped pkg_resources, which torch.utils.cpp_extension (used by
# detectron2's build) still imports; numpy must be <2 to match the ABI
# torch==2.1.0 was compiled against, or importing torch during that same build
# crashes with a NumPy 1.x/2.x mismatch.
pip install "setuptools<81" numpy==1.26.4

# Install Hamer
cd submodules/phantom-hamer
# --no-build-isolation: see note above about detectron2 needing torch at build time.
pip install --no-build-isolation -e .\[all\]
pip install -v -e third-party/ViTPose
wget https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz
tar --warning=no-unknown-keyword --exclude=".*" -xvf hamer_demo_data.tar.gz
cd ../..

# Install mmcv
pip install mmcv==1.3.9
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install numpy==1.26.4

# Install phantom-robosuite
cd submodules/phantom-robosuite
pip install -e .
cd ../..

# Install phantom-robomimic
cd submodules/phantom-robomimic
pip install -e .
cd ../..

# Install additional packages
pip install joblib mediapy open3d pandas
# zarr: on-disk format used by generate_synthetic_pickplace.py's ReplayBuffer
# output (matches diffusion_policy's data layout).
pip install zarr==2.17.2 numcodecs==0.12.1
pip install transformers==4.42.4
pip install PyOpenGL==3.1.4
pip install Rtree
pip install git+https://github.com/epic-kitchens/epic-kitchens-100-hand-object-bboxes.git
pip install protobuf==3.20.0
pip install hydra-core==1.3.2
pip install omegaconf==2.3.0

# Download E2FGVI weights
cd submodules/phantom-E2FGVI/E2FGVI/release_model/
pip install gdown
# gdown>=6 dropped --fuzzy: it now auto-parses share URLs without it.
gdown https://drive.google.com/file/d/10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3/view?usp=sharing
cd ../..

# Install phantom-E2FGVI
pip install -e .
cd ../..

# Install phantom
pip install -e .

# Re-pin versions that get silently upgraded by robosuite/robomimic/opencv's own
# (unpinned) dependency resolution above:
#  - mujoco is never pinned elsewhere, so pip grabs whatever is latest at
#    install time; this robosuite fork (1.4.1) breaks on versions much newer
#    than 3.1.x (a joint-type numpy/enum comparison bug), and needs at least
#    the version that added the camera "sensorsize" attribute this pipeline's
#    camera calibration relies on -- 3.1.6 is a version that satisfies both.
#  - opencv-python's releases require numpy>=2 as of some point in its 4.x
#    line (a version *range* like "<5" is not enough -- even recent 4.x
#    releases now require numpy>=2), which breaks the ABI torch==2.1.0 was
#    compiled against (numpy<2). Pin an exact older opencv-python instead.
#  - numpy itself must come last: robosuite/robomimic/opencv-python's own
#    requirements all silently pull numpy back to 2.x if installed after this.
pip install mujoco==3.1.6
pip install opencv-python==4.9.0.80
pip install numpy==1.26.4

# Download sample data
mkdir -p data/raw
cd data/raw
wget https://download.cs.stanford.edu/juno/phantom/pick_and_place.zip
unzip pick_and_place.zip
rm pick_and_place.zip
wget https://download.cs.stanford.edu/juno/phantom/epic.zip
unzip epic.zip
rm epic.zip
cd ../..
