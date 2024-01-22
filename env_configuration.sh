wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh

conda create -n t2p python=3.8 numpy
conda create -n mask3d python=3.8 numpy

conda activate t2p

pip install torch torchvision torchaudio

#geometrics

pip install easydict

pip install scikit-learn
pip install matplotlib
pip install opencv-python
pip install plyfile
pip install open3d


conda create -n t2p_m python=3.8 numpy=1.20.3

conda activate t2p_m

pip install torch torchvision torchaudio

#geometrics

pip install easydict

pip install scikit-learn
pip install matplotlib
pip install opencv-python
pip install plyfile
pip install open3d