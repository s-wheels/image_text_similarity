# -*- coding: utf-8 -*-

import torch
from torchnlp.word_to_vector import GloVe
import pandas as pd
from sklearn.decomposition import TruncatedSVD


def main(labels_file="../data/flickr30k_images/results.csv",
         output_file="comment_embeddings/comment_embeddings_pc.pt"):
    """
    Takes in Flickr30k labels and computes comment embeddings
    using GloVe for word embeddings and the SIF weighted to transform
    these to a comment embedding.

    Parameters
    ----------
    labels_file : string, optional
        Location of flickr30k labels. The default is "../data/flickr30k_images/results.csv".

    """
    #Load up the csv file with labels
    labels_df = pd.read_csv(labels_file, sep="|")
    labels_df = preprocess_comments(labels_df)
    comment_lists, flat_comment_srs = create_flat_comments_series(labels_df)
    
    #Determine the corpus and counts for each word
    corpus_vocab = sorted(flat_comment_srs.unique())
    corpus_count = flat_comment_srs.value_counts()
    total_words = sum(corpus_count)
    
    #Download the GloVe embeddings
    glove_embedding = GloVe(name="6B", dim=100, is_include=lambda w: w in set(corpus_vocab))
    
    #Attempt to find embeddings for every word in the corpus
    vocab_embeddings = torch.Tensor(len(corpus_vocab), glove_embedding.dim)
    for i, token in enumerate(corpus_vocab):
        vocab_embeddings[i] = glove_embedding[token]
        
    corpus_vocab_refined = []
    vocab_embeddings_refined = []
    
    #Remove words from corpus if no embedding found
    for k, word_embedding in enumerate(vocab_embeddings):
        if len(word_embedding.nonzero()) != 0:
            vocab_embeddings_refined.append(word_embedding)
            corpus_vocab_refined.append(corpus_vocab[k])        
            
    vocab_embeddings = torch.stack(vocab_embeddings_refined)
    corpus_vocab = pd.Series(corpus_vocab_refined)
    corpus_vocab_count = pd.Series(data=corpus_vocab.map(corpus_count).values, index=corpus_vocab.values)
    corpus_vocab_prob = corpus_vocab_count/sum(corpus_vocab_count)    
    
    comment_embeddings = sif_embeddings(comment_lists, corpus_vocab_count, corpus_vocab_prob, vocab_embeddings)
    comment_embeddings_pc = remove_pc(comment_embeddings)
    torch.save(comment_embeddings_pc, output_file)


def preprocess_comments(input_str, input_type='dataframe'):
    """

    Parameters
    ----------
    labels_df : pandas.DataFrame
        DataFrame containing column with raw comments.

    Returns
    -------
    labels_df : pandas.DataFrame
        DataFrame containing column with comments processed to work with GloVe embeddings.

    """
    alter_strings = ["blond-hair", "red-hair", "short-sleeve", "long-sleeve", "?", "+", "(", ")"]
    replace_strings = ["blond hair", "red hair", "short sleeve", "long sleeve", "", "", "", ""]
    
    if input_type == 'dataframe':

        #Pre-process comments before input to GloVe
        input_str["comment"] = input_str["comment"].str.lstrip().str.rstrip(" .").str.lower()

        for i in range(len(replace_strings)):
            input_str["comment"] = input_str["comment"].str.replace(alter_strings[i], replace_strings[i])
            
    else:
        input_str = input_str.lower().replace(' .', '').strip()
        for i in range(len(replace_strings)):
            input_str = input_str.replace(alter_strings[i], replace_strings[i])

    return input_str


def create_flat_comments_series(labels_df):
    """

    Parameters
    ----------
    labels_df : pandas.DataFrame
        DataFrame containing column with comments.

    Returns
    -------
    pandas.Series
        Series containing comments in flat format.

    """
    
    
    #Split up words in comments and filter out empty strings
    comment_lists = list(labels_df["comment"].str.split(" ").values)
    #Create a nested list for comments
    flat_comment_list = []
    comment_lists_refined = []
    for sublist in comment_lists:
        comment_lists_refined.append(list(filter(lambda x: x != "", sublist)))
        for item in sublist:
            flat_comment_list.append(item)
    comment_lists = comment_lists_refined
    
    return comment_lists, pd.Series(flat_comment_list)

def compute_pc(X,npc=1):
    """
    Function taken from: https://github.com/PrincetonML/SIF

    Parameters
    ----------
    X : numpy.array
        Array containing SIF weighted embeddings of shape
        [number of comments, embedding size]
    npc : int, optional
        How many principle components to remove. The default is 1.

    Returns
    -------
    numpy.array
        Array of principle components.

    """

    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_

def remove_pc(X, npc=1):
    """
    Function modified from https://github.com/PrincetonML/SIF

    Parameters
    ----------
    X : torch.Tensor
        Tensor containing SIF weighted embeddings of shape
        [number of comments, embedding size].
    npc : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    torch.Tensor
        Tensor containing SIF weighted embeddings with principal components
        removed of shape [number of comments, embedding size].

    """
  
    X = np.array(X)
    pc = compute_pc(X, npc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return torch.Tensor(XX)

def sif_embeddings(comment_lists, corpus_vocab_count, corpus_vocab_prob, vocab_embeddings, alpha=1e-3):
    """
    Compute the SIF weighted embeddings for a list of comments

    Parameters
    ----------
    comment_lists : list
        DESCRIPTION.
    corpus_vocab_count : pandas.Series
        DESCRIPTION.
    corpus_vocab_prob : Pandas.Series
        DESCRIPTION.
    vocab_embeddings : torch.Tensor
        DESCRIPTION.
    alpha : float, optional
        DESCRIPTION. The default is 1e-3.

    Returns
    -------
    torch.Tensor
        Tensor containing SIF weighted comment embeddings of shape
        [number of comments, embedding size].

    """

    embedding_size = vocab_embeddings.shape[1]  # Embedding size
        
    comment_embeddings = torch.zeros([1, 100])
    
    total_comments = len(comment_lists)
    # Iterate all sentences
    for comment_number, comment in enumerate(comment_lists):
        word_count = 0
        comment_embedding = torch.zeros([embedding_size], dtype=torch.float16) # Summary vector
        
        for word in comment:
            word_in_corpus = corpus_vocab_count.index.str.match(word + "$").any()

            if word_in_corpus:
                try:
                    word_embedding = vocab_embeddings[corpus_vocab_count.index.get_loc(word)].clone()
                    comment_embedding = torch.add(comment_embedding,(alpha / (alpha + corpus_vocab_prob[word])) * word_embedding)
                    word_count += 1
                except:
                    print("Word is not in corpus. REGEX handling error. WORD: ", word)
                    print("-" *50)
                
        if word_count > 0:
            torch.div(comment_embedding,word_count)

        comment_embeddings = torch.cat([comment_embeddings, comment_embedding.reshape([1, embedding_size])], 0)
        
        if (comment_number % 10000) == 0:
            print("Percent Done:", 100*comment_number/total_comments, "%")
            print("-" *70)

            torch.save(comment_embeddings, "comment_embeddings.pt")
    
    torch.save(comment_embeddings, "comment_embeddings.pt")

    return comment_embeddings[1:]


if __name__ == "__main__":
    main()