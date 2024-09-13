# Temporal-Consistent-RGBT-Segmentation
The official implementation of **Temporal Consistency for RGB-Thermal Data-based Semantic Scene Understanding**. ([IEEE RA-L](https://ieeexplore.ieee.org/document/10675452)).

<div align=center>
<img src="https://github.com/lab-sun/Temporal-Consistent-RGBT-Segmentation/blob/main/docs/overview.png" width="900px"/>
</div>

## Introduction
We propose a temporally consistent framework for RGB-T semantic segmentation, which includes a method to synthesize images for the next moment and loss functions to ensure segmentation consistency across different frames.

## Dataset
Download [MF dataset](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/) or [Our preprocessed version](https://drive.google.com/file/d/1NFdIigejYmCHFrdN2MSe1vxHHAs739SA/view?usp=sharing) and place them in 'datasets' folder in the following structure:

```shell
<datasets>
|-- <MFdataset>
    |-- <RGB>
    |-- <Thermal>
    |-- <Label>
    |-- train.txt
    |-- val.txt
    |-- test.txt
```

## Pretrained weights
Download the pretrained segformer here [pretrained segformer](https://drive.google.com/drive/folders/10XgSW8f7ghRs9fJ0dE-EV8G2E_guVsT5?usp=sharing) and place them in 'pretrained' folder in the following structure:

```shell
<pretrained>
|-- <mit_b0.pth>
|-- <mit_b1.pth>
|-- <mit_b2.pth>
|-- <mit_b3.pth>
|-- <mit_b4.pth>
|-- <mit_b5.pth>
```

## Usage
* Clone this repo
```
$ git clone https://github.com/lab-sun/Temporal-Consistent-RGBT-Segmentation.git
```
* Build docker image
```
$ cd ~/Temporal-Consistent-RGBT-Segmentation
$ docker build -t docker_image_tcfusenet .
```
* Download the dataset
```
$ (You should be in the Temporal-Consistent-RGBT-Segmentation folder)
$ mkdir ./datasets
$ cd ./datasets
$ (download our preprocessed dataset.zip in this folder)
$ unzip -d .. dataset.zip
```
* To reproduce our results, you need to download our pretrained weights
```
$ (You should be in the Temporal-Consistent-RGBT-Segmentation folder)
$ mkdir ./pretrained
$ cd ./pretrained
$ (download the pretrained segformer weights in this folder)
$ mkdir ./TCFuseNet_pth
$ cd ./TCFuseNet_pth
$ (download our pretrained CMX weights in this folder)
$ docker run -it --shm-size 8G -p 1234:6006 --name docker_container_tcfusenet --gpus all -v ~/Temporal-Consistent-RGBT-Segmentation:/workspace docker_image_tcfusenet
$ cd /workspace
$ python3 eval.py
```
* To train our method
```
$ (You should be in the Temporal-Consistent-RGBT-Segmentation folder)
$ docker run -it --shm-size 8G -p 1234:6006 --name docker_container_tcfusenet --gpus all -v ~/Temporal-Consistent-RGBT-Segmentation:/workspace docker_image_tcfusenet
$ (currently, you should be in the docker)
$ cd /workspace
$ python3 train.py
```
* To see the training process
```
$ (fire up another terminal)
$ docker exec -it docker_container_igfnet /bin/bash
$ cd /workspace
$ tensorboard --bind_all --logdir=./runs/tensorboard_log/
$ (fire up your favorite browser with http://localhost:1234, you will see the tensorboard)
```

## Result
We offer the pre-trained weights of our method modified based on CMX and RTFNet.

### CMX
| Architecture | Backbone | mIOU | Weight |
|:---:|:---:|:---:|:---:|
| Our-dice | MiT-B2 | 60.8% | [CMX-B2-dice](https://drive.google.com/file/d/1lEjaTVDJCRhgYTWQuVPq9jLW5xF0YhBI/view?usp=sharing) |
| Our-con | MiT-B2 | 59.9% | [CMX-B2-con](https://drive.google.com/file/d/1ZxI-aGzot4WXw5TqxAibTEmlgZ7gSCYE/view?usp=sharing) |
| Our-con-acc | MiT-B2 | 60.0% | [CMX-B2-con-acc](https://drive.google.com/file/d/128QDv_XLIDEWS4G4hgIMbLrUF6OVzgHo/view?usp=sharing) |

### RTFNet
| Architecture | Backbone | mIOU | Weight |
|:---:|:---:|:---:|:---:|
| Our-con | ResNet-50 | 53.3% | [RTF-50-con](https://drive.google.com/file/d/1ymLvnk7HaKkisZOaafjz5P8GHNPcn7_2/view?usp=sharing) |
| Our-con | ResNet-152 | 54.1% | [RTF-152-con](https://drive.google.com/file/d/1-PuiNRB-bYRh7eiurUvgLnWSAepgBwNv/view?usp=sharing) |

## Citation

## Acknowledgement
Some of the codes are borrowed from [CMX](https://github.com/huaaaliu/RGBX_Semantic_Segmentation).
