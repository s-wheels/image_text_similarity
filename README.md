# Image Text Similarity

## Installation
Install the environment in the pytorch-110.yml file:
```
conda env create -f pytorch-110.yml --name img_txt_sim
```
To compute the similarity run:
```
python3 predict.py
```
Images/Strings can be changed within the script

## Report

The approach taken in this respository is based upon that in [1].

First, precomputed image features and precomputed text features are computed.

* The image features were computed using ResNet50 which had been pretrained on ImageNet [2]. The last layer of the network was removed to allow the information contained in the image features before the FC classification layers to be used by the similarity network.

* The text features were computer using GloVe embeddings [3] which were then converted to sentence embeddings using the method described in [4].

Second, the similarity network was trained using bidirectional ranking loss on triplets to increase the cosine similarity between positive pairs over negative pairs. This created a shared embedding space into which both the image and text features were transformed using the method in [1].

### Similarity Network

- INPUTS: Pretrained Image and Sentence Features
  - 2 Fully Connected Layers per input.
  - Uses ReLU after the first FC layer to capture non-linearity.
  - Batch Norm after the second FC layer, to regularise and stabilise training.
  - Normalisation after the second output layer which enforces equivalence between the Euclidean distance and Cosine similarity.
 
- OUTPUTS: Two embeddings one of the image features and one of the sentence features.

- LOSS FUNCTION: Bidirectional ranking loss on the cosine similarity between the two embeddings. (Enforces that positive pairs should have higher similarity than negative pairs).

- OPTIMISER: Stochastic Gradient Descent is used for simplicity and produces good enough results.


### References:

* [1] Liwei Wang, Yin Li, Jing Huang, Svetlana Lazebnik. Learning Two-Branch Neural Networks for Image-Text Matching Tasks. IEEE Transactions on Pattern Analysis and Machine Intelligence. 2018.

* [2] Kaiming He, Xiangqu Zhang, Shaoqing Ren, Jian Sun. Deep residual learning for image recognition. Microsoft Research. 2015.

* [3] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. GloVe: Global Vectors for Word Representation. 2014.

* [4] Sanjeev Arora, Yingyu Liang, Tengyu Ma. A Simple but Tough to Beat Baseline for Sentence embeddings. ICLR, 2017.


## Challenges

One of the major issues in the ability for the network to derive context. The network does not take into account localisation information for images, as the pretrained ResNet was trained solely for image classification. To allow this to happen, the pretrained image features would need to be changed. Initially I would like to try feeding in segmentation maps to the network, although this would require a more complex network to deal with the larger information available to it.
