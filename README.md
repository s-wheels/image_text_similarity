# image_text_similarity


Dependencies:
torch
from torchvision import transforms
from embedding_model import create_models
from embed_comments import preprocess_comments

from PIL import Image

from torchnlp.encoders.text import WhitespaceEncoder
from torchnlp.word_to_vector import GloVe

import pandas as pd

References:

* Wang, Liwei
Li, Yin
Huang, Jing
Lazebnik, Svetlana. Learning Two-Branch Neural Networks for Image-Text Matching Tasks. IEEE Transactions on Pattern Analysis and Machine Intelligence

* Arora, Sanjeev
Liang, Yingyu
Ma, Tengyu. A simple but Tough to Beat Baseline for Sentence embeddings. ICLR, 2016.

* Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. [pdf] [bib]


* He, Kaiming
Zhang, Xiangqu
Ren, Shaoqing
Sun, Jian. Deep residual learning for image recognition. Microsoft Research.
