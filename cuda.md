# Install Nvidia Driver
sudo apt-get install make gcc -y
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/515.76/NVIDIA-Linux-x86_64-515.76.run
sudo bash NVIDIA-Linux-x86_64-515.76.run

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda-repo-ubuntu2004-11-7-local_11.7.1-515.65.01-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-7-local_11.7.1-515.65.01-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Environment variables
sudo nano ~/.bashrc
export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/include:$LD_LIBRARY_PATH

# Install cuDNN
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get install libcudnn8=8.5.0.*-1+cuda11.7
sudo apt-get install libcudnn8-dev=8.5.0.*-1+cuda11.7

# Verify
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python3 -c "import torch; print(torch.cuda.is_available())"

# Install with Docker
## Dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
## Create
nvidia-docker build -t nghiaht5 .
nvidia-docker run -it --name nghiaht5_con nghiaht5
python -c "import torch; print(torch.cuda.is_available())"
