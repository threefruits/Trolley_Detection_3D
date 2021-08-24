this is a interface for trolley pose detection.

# Get Start
---
Miniconda, Git install is required.

    git clone git@github.com:threefruits/Trolley_Detection_3D.git

'Notice: python=3.6'

    conda create -n Air python=3.6

    conda activate Air

    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

    conda install cudatoolkit=10.2

Start a new terminal run:

    roslaunch realsense2_camera rs_camera.launch

Return to conda terminal:

    python ros_interface.py