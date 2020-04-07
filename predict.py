from __future__ import print_function, division

import torch
from torchvision import transforms
from embedding_model import create_models
from embed_comments import preprocess_comments

from PIL import Image

from torchnlp.encoders.text import WhitespaceEncoder
from torchnlp.word_to_vector import GloVe

import pandas as pd

def main():
    # String must have spaces between words and punctuation
    string = 'A dog on grass .'
    img_name = 'images/dog.jpg'
    model_path = 'model_backups/similarity_model_final.pt'

    txt_features = get_comment_embed(string)


    resnet_model, similarity_model, device = create_models()

    test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    img = Image.open(img_name)
    img = test_transform(img)

    img_features = resnet_model(img.reshape(1,3,224,224))
    img_features = img_features.reshape(1, 2048)
    txt_features = txt_features.reshape(1,100)

    similarity_model.load_state_dict(torch.load(model_path, map_location=device))
    similarity_model.to(device)
    similarity_model.eval()
    img_embed, txt_embed = similarity_model(img_features, txt_features)

    similarity_score = torch.matmul(txt_embed, img_embed.t())
    print("Image: ", img_name, " | Text: ", string, " | Similarity: ", similarity_score.item())
    
    return similarity_score


def get_comment_embed(string, corpus_vocab_prob_file='pandas_objects/corpus_vocab_prob.pkl'):

    string_list = preprocess_comments(string, input_type='string').split(" ")
    string_list = list(filter(lambda x: x != "", string_list))
    string_list
    glove_embedding = GloVe(name="6B", dim=100, is_include=lambda w: w in set(string_list))

    corpus_vocab_prob = pd.read_pickle(corpus_vocab_prob_file)

    comment_embedding = torch.zeros([100]) # Summary vector

    for word in string_list:
        word_embedding = glove_embedding[word]
        try:
            word_prob = corpus_vocab_prob[word]
            comment_embedding = comment_embedding + (1e-3 / (1e-3 + word_prob) * word_embedding)
        except:
            print('Word not in Flickr Corpus. WORD: ', word)

    return comment_embedding


if __name__ == "__main__":
    main()

