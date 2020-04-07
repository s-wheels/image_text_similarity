from __future__ import print_function, division

import os
import time
import subprocess

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from embedding_model import create_models
from PIL import Image


def main(epochs=20, batch_size=10):


    # Get the Resnet model for image features and the Similarity model for embedding
    resnet_model, similarity_model, device = create_models()
    optimizer = torch.optim.SGD(similarity_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.794)

    comment_batch = batch_size * 5

    # Create image transforms for augmentation and processing by Resnet
    train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(), 
            transforms.Resize(224),             
            transforms.CenterCrop(224),         
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], #Normalisation values for ImageNet dataset
                                 [0.229, 0.224, 0.225])
        ])

    test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    # Build the datasets and dataloaders for [images, comments] for both [train, test]

    flickr_train_image_dataset = Flickr30kImageDataset(labels_file="labels_train_df.pkl",
                                                       img_dir="../data/flickr30k_images/train",
                                                       transform=train_transform)

    flickr_train_image_dataloader = DataLoader(flickr_train_image_dataset,
                                               batch_size=batch_size,
                                               num_workers=batch_size,
                                               drop_last=True)

    flickr_train_comment_dataset = Flickr30kCommentDataset(labels_file="labels_train_df.pkl",
                                                           txt_features_file="comment_train_features_pc.pt")

    flickr_train_comment_dataloader = DataLoader(flickr_train_comment_dataset,
                                                 batch_size=comment_batch,
                                                 num_workers=batch_size,
                                                 drop_last=True)

    flickr_test_image_dataset = Flickr30kImageDataset(labels_file="labels_test_df.pkl",
                                                       img_dir="../data/flickr30k_images/test",
                                                       transform=test_transform)

    flickr_test_image_dataloader = DataLoader(flickr_test_image_dataset,
                                              batch_size=batch_size,
                                              num_workers=batch_size,
                                              drop_last=True)

    flickr_test_comment_dataset = Flickr30kCommentDataset(labels_file="labels_test_df.pkl",
                                                           txt_features_file="comment_test_features_pc.pt")

    flickr_test_comment_dataloader = DataLoader(flickr_test_comment_dataset,
                                                batch_size=comment_batch,
                                                num_workers=batch_size,
                                                drop_last=True)

    num_batches = len(flickr_train_image_dataloader)
    num_test_batches = len(flickr_test_image_dataloader)
    print_loss = num_batches // 5
    train_losses = []
    test_losses = []

    # Create booleans for calculation of ranking loss and send to same device as model
    pos_triplet_bool, neg_triplet_bool = get_triplet_bool(batch_size)
    pos_triplet_bool = pos_triplet_bool.to(device)
    neg_triplet_bool = neg_triplet_bool.to(device)

    for i_epoch in range(epochs):
        epoch_start_time = time.time()

        scheduler.step()
        txt_enumeration = enumerate(flickr_train_comment_dataloader)
        accumlated_train_loss = 0
        accumlated_test_loss = 0
        for i_batch, imgs_batched in enumerate(flickr_train_image_dataloader):

            #Get the pretrained image (resnet) and text (glove SIF) features
            with torch.no_grad():
                img_features = resnet_model(imgs_batched)
                img_features = img_features.reshape(-1, 2048)
                txt_features = next(txt_enumeration)[1]

            # Apply the model
            img_embed, txt_embed = similarity_model(img_features, txt_features)

            # Calculate the bidirectional ranking loss for positive and negative pairs.
            loss = bidirectional_ranking_loss(img_embed, txt_embed,
                                              pos_triplet_bool, neg_triplet_bool, batch_size)        

            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print interim results
            if i_batch % print_loss == 0:

                print("Epoch: ", i_epoch, " | Batch:", i_batch, "/", num_batches, " | Loss: ", loss.item())

            accumlated_train_loss += loss

        # Save model checkpoint
        model_path = "model_backups/similarity_model_e" + str(i_epoch) + ".pt"
        print("Saving model backup: ", model_path)
        torch.save({"epoch": i_epoch,
                   "model_state_dict": similarity_model.state_dict(),
                   "optimizer_state_dict": optimizer.state_dict(),
                   "loss": accumlated_train_loss}, model_path)


        train_losses.append(accumlated_train_loss)

        # Run the testing batches
        with torch.no_grad():
            txt_enumeration = enumerate(flickr_test_comment_dataloader)
            for i_batch, imgs_batched in enumerate(flickr_test_image_dataloader):

                img_features = resnet_model(imgs_batched)
                img_features = img_features.reshape(-1, 2048)
                txt_features = next(txt_enumeration)[1]

                # Apply the model
                img_embed, txt_embed = similarity_model(img_features, txt_features)
                loss = bidirectional_ranking_loss(img_embed, txt_embed,
                                                  pos_triplet_bool, neg_triplet_bool, batch_size)

                accumlated_test_loss += loss

            test_losses.append(accumlated_test_loss)

        print("-" * 50)
        print("Duration of epoch: ", time.time() - epoch_start_time, "seconds") # print the time elapsed
        print("Mean train loss: ", accumlated_train_loss.item() / num_batches)
        print("Mean test loss: ", accumlated_test_loss.item() / num_test_batches)
        print("-" * 50)


    final_model_path = "model_backups/similarity_model_final.pt"
    print("Saving final model: ", final_model_path)
    torch.save(similarity_model.state_dict(), final_model_path) 



def get_triplet_bool(batch_size):
    
    pos_triplet_boolean_mask = torch.zeros([batch_size, batch_size*5])

    for i in range(batch_size):
        pos_start = i * 5
        pos_end = pos_start + 5
        for j in range(pos_start,pos_end):
            pos_triplet_boolean_mask[i,j]=1
    
    pos_triplet_boolean_mask = pos_triplet_boolean_mask.t()
    neg_triplet_boolean_mask = (pos_triplet_boolean_mask * -1) + 1
    
    return pos_triplet_boolean_mask.byte(), neg_triplet_boolean_mask.byte()

def pdist_loss(img_embed, txt_embed):
    x1_sq = torch.sum(txt_embed * txt_embed,dim=1).reshape([-1, 1])
    x2_sq = torch.sum(img_embed * img_embed,dim=1).reshape([1, -1])
    cos_sim = torch.matmul(txt_embed, img_embed.t())
    return torch.sqrt(x1_sq - 2 * cos_sim + x2_sq)

def bidirectional_ranking_loss(img_embed, txt_embed,
                               pos_triplet_bool, neg_triplet_bool,
                               batch_size, img_loss_factor=1.5, margin=0.05):
    
    cos_sim = pdist_loss(img_embed, txt_embed)
    
    max_k = min(10, batch_size-1)
    
    #Image loss
    pos_pair_dist = torch.masked_select(cos_sim, pos_triplet_bool).reshape([batch_size*5,1])
    neg_pair_dist = torch.masked_select(cos_sim, neg_triplet_bool).reshape([batch_size*5,-1])

    img_loss = torch.clamp(margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
    img_loss  = torch.topk(img_loss, max_k)[0].mean()

    neg_pair_dist = torch.masked_select(cos_sim.t(), neg_triplet_bool.t()).reshape([batch_size, -1])
    neg_pair_dist = neg_pair_dist.repeat([1,5]).reshape([batch_size * 5, -1])

    sent_loss = torch.clamp(margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
    sent_loss  = torch.topk(sent_loss, max_k)[0].mean()

    loss = (img_loss_factor * img_loss) + sent_loss
    
    return loss


class Flickr30kImageDataset(Dataset):
    """Flickr30K Image dataset"""
    
    def __init__(self, labels_file, img_dir, transform=None):
        self.images_df = pd.read_pickle(labels_file).groupby("image_name").count()
        self.img_dir = img_dir
        self.transform = transform
                
    def __len__(self):
        return len(self.images_df)
    
    def __getitem__(self, idx):
        
        img_name = os.path.join(self.img_dir, self.images_df.index.values[idx])
        img = Image.open(img_name)
        
        if self.transform:
            img = self.transform(img)
        
        return img        
    
    
class Flickr30kCommentDataset(Dataset):
    """Flickr30K Comment Features dataset"""
    
    def __init__(self, labels_file, txt_features_file):
        self.labels_df = pd.read_pickle(labels_file)
        self.txt_features = torch.load(txt_features_file)
        
        assert len(self.txt_features) == len(self.labels_df), "Precomputed text embeddings do not match labels file"
        
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        
        txt_features = self.txt_features[idx]
                
        return txt_features       


if __name__ == "__main__":
    main()