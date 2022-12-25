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

`$ python monoscene/data/semantic_kitti/preprocess.py kitti_root= /path/to/kitti/preprocess/folder kitti_preprocess_root=/path/to/semantic_kitti`

# 2.KITTI Object Detection Dataset

1. Download the 3D KITTI detection dataset from [here](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). Download Velodyne Point Clouds, Training labels of object data set, Camera calibration matrices of object dataset, and left color images of object data set. 

2. In Complex-YOLOv4 dataset, create two folders train and test folders inside the kitti folder. Copy the downloaded image_2, calib, label and velodyne folders from the downloaded folders into train folder. For test folder, only copy the calib folder, while copying the testing images and velodyne folder from the Semantic Kitti downloaded dataset. 


# Running the models

# 1. Training Monoscene
1. Inside Semantic Kitti folder, create folders to store training logs at /path/to/kitti/logdir.

2. Run the following command to train the monoscene model. 

`$ cd Semantic Segmentation and Object Detection/`

`$ python monoscene/scripts/train_monoscene.py +dataset=kitti +enable_log=true +    kitti_root=/path/to/semantic_kitti  +kitti_preprocess_root=/path/to/kitti/preprocess/folder +    kitti_logdir=/path/to/kitti/logdir +n_gpus=4 +batch_size=4`


# 2. Training Complex YOLOv4
To train Complex YOLOv4, run the following commands. 
`cd Complex-YOLOv4/src'

`python train.py --gpu_idx 0 --batch_size 4 --num_workers 1`


# 3. Evaluating Monoscene 
`cd Semantic Segmentation and Object Detection/`
`python monoscene/scripts/eval_monoscene.py +dataset=kitti  +    kitti_root=/path/to/semantic_kitti  +kitti_preprocess_root=/path/to/kitti/preprocess/folder +n_gpus=1 +batch_size=1`

# 4. Testing Complex YOLOv4
`python test.py --gpu_idx 0 --pretrained_path ../checkpoints/complex_yolov4/complex_yolov4_mse_loss.pth --cfgfile ./config/cfg/complex_yolov4.cfg --show_image`


# Sample Results

![sample_result_1](https://user-images.githubusercontent.com/80807952/209465690-3d813521-66ac-41a8-ab75-842ec0a79014.png)

![sample_result_2](https://user-images.githubusercontent.com/80807952/209465692-b192aa9f-7c33-4dbe-8a75-b91f8a7b0622.png)


![sample_result_3](https://user-images.githubusercontent.com/80807952/209465697-b51020c3-7a6a-4a04-b9e1-351228a767be.png)
