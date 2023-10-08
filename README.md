# cuda_install

### Overview Installing

    • Cuda 10.0.130_410.48
    • Gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
    • Nvidia-driver 450.80.02
    • Cudnn 7_7.5.1.10
    • Python 3.7


### Installing

#### 1. Remove old cuda/nvidia environment:
    $ sudo apt-get --purge -y remove 'cuda*'
    $ sudo apt-get --purge -y remove 'nvidia*'
    $ sudo reboot

	or

	sudo apt-get purge nvidia*
	sudo apt-get autoremove
	sudo apt-get autoclean
	sudo rm -rf /usr/local/cuda-10.0

#### 2.Update and Upgrade :
    $ sudo apt-get update
    $ sudo apt-get upgrade


#### 3.Install Nvidia Drivers
    • Download driver from :
                                https://www.nvidia.in/Download/index.aspx?lang=en-in
    $ chmod +x NVIDIA-Linux-x86_64-410.93.run
    $ nvidia-smi
    $ to remove :sudo apt-get purge nvidia-*

	or:
	sudo apt-get purge nvidia-*
	sudo add-apt-repository ppa:graphics-drivers
	sudo apt-get update
	sudo apt-get install nvidia-381


#### 4.Install CUDA
    • Download the runfile (local) from : https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1710&target_type=runfilelocal

    $ chmod +x ./cuda_10.0.130_410.48_linux.run
    $ sudo ./cuda_10.0.130_410.48_linux.run
    $ nvcc -V


#### 5.Set cuda/LD_LIBRARY_PATH Paths
    $ export PATH="/home/nghiaht5/anaconda3/bin:/usr/local/cuda/bin:$PATH"
    $ export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    $ source ~/.bashrc
    $ nano /home/$USER/.bashrc (check again) or gedit ~/.bashrc

#### 6.Install Cudnn
    • Download CUDA from : https://developer.nvidia.com/cudnn
    • Download all 3 .deb files: the runtime library, the developer library, and the code samples library for Ubuntu 18.04.
    run:
       $ sudo dpkg -i libcudnn7_7.5.1.10-1+cuda10.0_amd64.deb
       $ sudo dpkg -i libcudnn7-dev_7.5.1.10-1+cuda10.0_amd64.deb
       $ sudo dpkg -i libcudnn7-doc_7.5.1.10-1+cuda10.0_amd64.deb

#### 7.MNIST example code
     $ cd /usr/src/cudnn_samples_v7/mnistCUDNN/
     $ sudo make clean && sudo make
     $ ./mnistCUDNN
     Now you see : Test passed!

#### 8.Verify
    $ python3
    $ import tensorflow as tf
    $ sess = \
    ... tf.Session(config=tf.ConfigProto(log_device_placement=True))


### Compile pointconv
#### 1.compile
    • From 3 sh file .sh to change : CUDA_PATH
    • Check
        print(tf.sysconfig.get_lib())
        --->  /home/nuptn1/.local/lib/python3.6/site-packages/tensorflow

    • Change: libtensorflow_framework.so.2 to libtensorflow_framework.so
    • Fix tf1.13.1 error:
        find your CUDA install path, in my case it is /usr/local/cuda
        $ export LD_LIBRARY_PATH=/usr/local/cuda/lib64
        $ source ~/.bashrc
        Then TF follows LD_LIBRARY_PATH to locate libcublas.so.10.0
        
#### 2.Setup to run in cuda env
    $ export PATH="/home/nghiaht5/anaconda3/bin:/usr/local/cuda/bin:$PATH"
    $ export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    $ export LD_LIBRARY_PATH=/usr/local/cuda/lib64
    $ source ~/.bashrc
    
    
### Recommendation installing:
#### 1.Add NVIDIA package repositories
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
        sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
        sudo dpkg -i cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
        sudo apt-get update
        wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
        sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
        sudo apt-get update

#### 2.Install NVIDIA driver
    sudo apt-get install --no-install-recommends nvidia-driver-450
#### 3.Reboot. Check that GPUs are visible using the command: nvidia-smi

#### 4.Install development and runtime libraries (~4GB)
    sudo apt-get install --no-install-recommends \
        cuda-10-1 \
        libcudnn7=7.6.5.32-1+cuda10.1  \
        libcudnn7-dev=7.6.5.32-1+cuda10.1


#### 5.Install TensorRT. Requires that libcudnn7 is installed above.
    sudo apt-get install -y --no-install-recommends libnvinfer6=6.0.1-1+cuda10.1 \
        libnvinfer-dev=6.0.1-1+cuda10.1 \
        libnvinfer-plugin6=6.0.1-1+cuda10.1

