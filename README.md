# Vision transformer

What we saw in class was that transformers were created for NLP tasks, and due to their outstanding performance, they have become the state of the art in this field. After the great success of transformers in the field of natural language processing, there have been many attempts to use this type of architecture in the field of computer vision, but some of them were limited to using another types of architecture, such as CNNs. However in 2021, the first pure vision transformer was known. Here is the [paper](https://arxiv.org/abs/2010.11929) and the [code](https://github.com/google-research/vision_transformer). So, first If you have some doubt about the vision transformer you can read [this post](https://medium.com/machine-intelligence-and-deep-learning-lab/vit-vision-transformer-cc56c8071a20).

<p align="center">
<img src="img/ViT.gif" width="75%">
</p>

In this task you are going to be using transformers to classify classes. This taks have two main parts. The first part is a practical activity and the second is a theoretical one. All the answer for the questions must be in your final report.

## Objectives
- To comprehend why the use of patches as a solution is useful for vision transformer.
- To understand the attention in transformer in a intuitive way.
- To know how the positional embedding works in vision transformers.

## Vision transformer Instructions

### Installing

Previously install the requirements and download the weights.
```
# Install pytorch librearies
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install other requirements
pip install -r requirements.txt

# imagenet21k pre-train + imagenet2012 fine-tuning
wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz
```

## Part 1: Transformer training (3.5 points)

Now let's implement a transformer model. As you can see we are going to be using ViT Base whose characteristics can be observed in the following table.

| Model        | Layers | Hidden size | D MLP size | Heads | Params  |
|--------------|--------|-------------|------------|-------|---------|
| ViT-Base    |   12   |     768     |    3072    |  12   |  86M    |
 
### 1.1 Divide an image into patches (1.25 points)

In this part you are going to implement the patch embedding which is a small function that is going to divide the image in to patches of 16*16 in this case. You are going to implement the missing code in the function embedding of the [modeling](models/modeling.py) file.

Why is important to divide the image into patches? why shouldn't we use each pixel value as a token?

After completing the missing part of the code, we can now proceed to train this model. To do this, you can use the following command:

```
python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir ViT-B_16-224.npz
```

Notice that the training phase can last 5 hours so plan ahead.

### 1.2 Experiments (1.25 point)

Now, we are going to do some experimentation, so you should conduct at least three experiments by varying the hyperparameters that interest you the most. Include a table and discuss the results in the report.

For this part you can decrease the number of steps of the model and use Automatic Mixed Precision(Amp) to train faster.

```
python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16-224.npz --fp16
```

### 1.3 Attention map (1 points)

In this section, we will explore one of the most critical aspects of the transformer model: attention maps. You will find a Jupyter notebook named [attention map](attention_map.ipynb) in which you are going to implement the code to visualize the attention map. Please follow the provided instructions to complete the missing code. Once done, upload the Jupyter notebook to your repository. You have to discuss the results and explain the process to get the attention map on a image and add it to your report.

#### Bonus (0.2)
Implement the attention map for every attention heads of the transformer.

#### Bonus (0.1)
Use your own images and visualize all the attention maps for each attention head and dicuss it in your report.

## Part 2: (1.5 points)

The main goal of this part is to answer some questions that are important to understand what is a transformer, what are the most important things and why we use it. You can refer to this [paper](https://arxiv.org/abs/2010.11929) to answer the questions.

- Short description of the main components of a Transformer architecture (max 2 paragraphs). 
- What are the most relevant hyperparameters for the Transformer strategy?
- How would changes in the parameters improve performance?
- What are the biggest differences between CNNs and Transformers beyond their architectures?



## Deadline and Report
Please upload to your repository a PDF file names Lastname_ViT.pdf. \
Deadline: Nov 6, 2023, 23:59

## References
[Vision transformer](https://arxiv.org/pdf/2010.11929.pdf)

## Sources
This repository is an adaptation from: \
[pytorch-image-models](https://github.com/huggingface/pytorch-image-models.git)\
[ViT Pytorch](https://github.com/jeonsworld/ViT-pytorch.git)
