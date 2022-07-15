from __future__ import division, print_function

import glob

import pandas as pd
import torch
from PIL import Image
from torchnlp.encoders.text import WhitespaceEncoder
from torchnlp.word_to_vector import GloVe
from torchvision import transforms

from embed_comments import preprocess_comments
from embedding_model import create_models

# String must have spaces between words and punctuation
strings = [
    "Children and kids having fun .",
    "Animals , like a horse , a dog or a cat .",
]
img_names = glob.glob("images/*.jpg")


def main(string: str, img_name: str, glove_embedding: GloVe) -> float:

    model_path = "model_backups/similarity_model_final.pt"

    txt_features = get_comment_embed(string, glove_embedding)

    # Load models for calculating image features and determining similarity
    resnet_model, similarity_model, device = create_models()

    # Define image transforms and apply to loaded image
    test_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open(img_name)
    img = test_transform(img)

    # Get img features and reshape img/txt features
    img_features = resnet_model(img.reshape(1, 3, 224, 224))
    img_features = img_features.reshape(1, 2048)
    txt_features = txt_features.reshape(1, 100)

    # Load up similarity model weights and evaluate inputs
    similarity_model.load_state_dict(torch.load(model_path, map_location=device))
    similarity_model.to(device)
    similarity_model.eval()
    img_embed, txt_embed = similarity_model(img_features, txt_features)

    similarity_score = torch.matmul(txt_embed, img_embed.t())
    print(
        "Image: ",
        img_name,
        " | Text: ",
        string,
        " | Similarity: ",
        similarity_score.item(),
    )

    return similarity_score.item()


def get_comment_embed(
    string: str,
    glove_embedding: GloVe = None,
    corpus_vocab_prob_file: str = "pandas_objects/corpus_vocab_prob.pkl",
) -> torch.Tensor:
    """

    Parameters
    ----------
    string : str
        Comment associated (or not) with image.
    glove_embedding: dict
    corpus_vocab_prob_file : str, optional
        File location of corpus vocab probability pickle file.
        The default is 'pandas_objects/corpus_vocab_prob.pkl'.

    Returns
    -------
    comment_embedding : torch.Tensor
        [1, 100]

    """

    string_list = preprocess_comments(string, input_type="string").split(" ")
    string_list = list(filter(lambda x: x != "", string_list))

    if glove_embedding == None:
        glove_embedding = GloVe(
            name="6B", dim=100, is_include=lambda w: w in set(string_list)
        )

    corpus_vocab_prob = pd.read_pickle(corpus_vocab_prob_file)

    comment_embedding = torch.zeros([100])  # Summary vector

    for word in string_list:
        word_embedding = glove_embedding[word]
        try:
            word_prob = corpus_vocab_prob[word]
            comment_embedding = comment_embedding + (
                1e-3 / (1e-3 + word_prob) * word_embedding
            )
        except:
            print("Word not in Flickr Corpus. WORD: ", word)

    return comment_embedding


if __name__ == "__main__":

    for string in strings:
        similarities = []
        string_list = preprocess_comments(string, input_type="string").split(" ")
        string_list = list(filter(lambda x: x != "", string_list))
        glove_embedding = GloVe(
            name="6B", dim=100, is_include=lambda w: w in set(string_list)
        )

        for img_name in img_names:
            similarities.append([img_name, main(string, img_name, glove_embedding)])

        similarities.sort(key=lambda x: x[1])
        similarities.reverse()

        print("From most similar to least similar...")
        for i in range(1, len(similarities) + 1):
            print("IMG: ", similarities[i - 1], "IMG RANK:", i)
