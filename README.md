# **DeepLabV3 implemetation**

This work aims to reproduce DeeplabV3 using different backbone by Pytorch.

Task: Semantic Segmentation

Paper: [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

## **Environment Preparation**

Python: 3.9

Run following command to install needed packages:

```bash
pip install -r requirements.txt
```

This work has already set up to train on [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/) (for segmentation task). If you are training/testing the model for the first time, you should download the dataset first.

To download dataset, you can modify the source code in main.py/predict.py or test.py file (VOCSegmentation(..., download=True,...), default: False).

- 2012_train: should be in "dataset/train" folder after downloaded.
- 2012_val: should be in "dataset/val" folder after downloaded.

It is highly recommend to use "2012_trainaug" - an expanded dataset based on "train" set of PASCAL VOC dataset, to get higher accuracy.

You can download [here](https://drive.google.com/file/d/1GR-dhj86rY8hVXUu3KTZ6vtd-sgT-Qzr/view?usp=sharing). After unzip the file, put:

- "train_aug.txt" file into "dataset/train/VOCdevkit" folder
- "SegmentationClassAug" folder into "dataset/train/VOCdevkit/VOC2012" folder

## **Training new model**

This is an example of training command:

```bash
python main.py 16 -lr 0.001 -e 200 -b 16 --trainaug --gpu-id 6 -v
```

Type the below command to know more arguments:

```bash
python main.py -h
```

The final checkpoints after training process can be found at "out/.../*pth" folder. There are two exported checkpoint files:

- best_params.pth: contain only the best value of parameters (which gave the highest mIoU)
- best_model.pth: contain both the best parameters and model dict

### **Visualize training process**

It is crucial to keep track of training process (loss, acc,...) to ensure your training setting is appropriate.

Use folloing command (in another terminal) to enable tensorboard:

```bash
tensorboard --logdir runs --port PORT
```

Tensorboard will be available at "http://localhost:PORT"

## **Validate model**

Run the following command to get mIoU and class IoU of a trained model (using checkpoint file at CP_PATH):

```bash
python test.py CP_PATH 8 --gpu-id 5 
```

To know more usefule arguments, use below command:

```bash
python test.py -h
```

## **Visualize some samples from validation set"

To be more aware of how well your model perform on data, you can use a simple command to print out some results from validation set:

```bash
python predict.py CP_PATH 16 --gpu-id 7 --num-samples 10
```

Also use argument "-h" to know more options.

We also provide some pretrained models, you can check out them [here](https://drive.google.com/drive/folders/1FxDBOgIpYUJrxB-4wTu0A60_BuchCXYR?usp=sharing)

## **Some results**

|  Model              | Batch Size  | train/val OS  |  mIoU     |
| :--------           | :---------: | :-----------: | :-------: |
| DeepLabV3_Reset50   | 16          |  16/16        |  0.7540   |
| DeepLabV3_ResNet101 | 16          |  16/16        |  0.7804   |

Inference results:

![alt text](https://lh3.googleusercontent.com/I6c8FcDCMXK_d4lWuuDCs7Cz2QxUSJXENDEWbsfnpOXpeISNIJ9KIidUkwIPwAPXHjs=w2400)
