# Semantic-Segmentation-and-Object-Detection-on-Point-Cloud-Data

This project is an implementation of Semantic Segmentation and Object Detection on Point Cloud Data, using the MonoScene and Complex YOLOv4 respectively. The datasets considered are SemanticKITTI and Kitti 3D Object Detection Datasets respectively. 

# Requirements
1. Install Anaconda 
2. Install CUDA 11.7
3. Please install Pytorch 13.1 with python 3, CUDA 11.7.

`$ conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`

4. For installing additional dependencies, change the directory into this repo, and install using the requirements.txt file. 

`$ cd Semantic Segmentation and Object Detection/'

'$ pip install -r requirements.txt`


5. Install tbb

`$ conda install -c bioconda tbb`

6. Downgrade torchmetrices to 0.6.0

`$ pip install torchmetrics==0.6.0`

7. Install Monoscene

`$ cd Monoscene`
`$ pip install -e ./`

8. Install mayavi and shapely libraries. 

# Datasets 

# 1.SemanticKitti Dataset

1. Download the Semantic Scene Completion Dataset from [here](http://www.semantic-kitti.org/dataset.html#download). Also, download the KITTI Odometry Benchmark Calibration data, and the RGB images from this [link](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).

2. Create a folder named kitti/preprocess/folder.

3. Preprocess the data using the following commands.

`$ cd MonoScene/`

`$ python monoscene/data/semantic_kitti/preprocess.py kitti_root= /path/to/kitti/preprocess/folder kitti_preprocess_root=/path/to/semantic_kitti

# 2.KITTI Object Detection Dataset

1. Download the 3D KITTI detection dataset from [here](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). Download Velodyne Point Clouds, Training labels of object data set, Camera calibration matrices of object dataset, and left color images of object data set. 

2. In Complex-YOLOv4 dataset, create two folders train and test folders inside the kitti folder. Copy the downloaded image_2, calib, label and velodyne folders from the downloaded folders into train folder. For test folder, only copy the calib folder, while copying the testing images and velodyne folder from the Semantic Kitti downloaded dataset. 







