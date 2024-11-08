from torchvision import transforms, datasets
import torch
import torch.optim as optim
import torch.nn.functional as F
import clip
from PIL import Image
from sklearn.metrics import accuracy_score, normalized_mutual_info_score as nmi, rand_score as ri, adjusted_rand_score as ar, f1_score as f1
from sklearn.cluster import KMeans
import numpy as np
import os
import random
from parse import get_parse
import time
import gc

gc.collect()
torch.cuda.empty_cache()

data_transfrom = transforms.Compose([
    transforms.ToTensor()
])


num_tokens = 77 # CLIP default token length

def encode_text_with_learnt_tokens(self, text, asterix_token, learnable_codes):
    """Generate proxy word embeddings using learned tokens."""
    placeholder_rows, placeholder_cols = torch.where(text == asterix_token)
    x = self.token_embedding(text).type(self.dtype)

    for i in range(len(placeholder_rows)):
        x_i_longer = torch.cat((x[placeholder_rows[i]][:placeholder_cols[i]], learnable_codes[i].unsqueeze(0), x[placeholder_rows[i]][placeholder_cols[i]+1:]), 0).to(learnable_codes.dtype)
        x[i] = x_i_longer[:num_tokens]

    x = x.permute(1, 0, 2)
    x = self.transformer(x)
    x = x.permute(1, 0, 2)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
    return x

def reference_word_embedding(self, text):
    embedding = []
    idx = torch.argmin(text, dim=1) - 1
    x = self.token_embedding(text).type(self.dtype)
    for i in range(len(text)):
        tmp = torch.mean(x[i][1:idx[i]], dim=0)
        embedding.append(tmp)
    return torch.stack(embedding)

# def clustering_quality_loss(embeddings, labels, margin=1.0, lambda_factor=0.5):
    # """Calculate clustering quality loss with intra- and inter-cluster components."""
    # distance_matrix = torch.cdist(embeddings, embeddings, p=2)

    # intra_cluster_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    # intra_cluster_distances = distance_matrix * intra_cluster_mask.float()
    # intra_cluster_loss = intra_cluster_distances.sum() / intra_cluster_mask.sum()

    # inter_cluster_mask = labels.unsqueeze(0) != labels.unsqueeze(1)
    # inter_cluster_distances = distance_matrix * inter_cluster_mask.float()
    # inter_cluster_loss = F.relu(margin - inter_cluster_distances).sum() / inter_cluster_mask.sum()

    # # Weighted combination of intra- and inter-cluster loss
    # loss = lambda_factor * intra_cluster_loss + (1 - lambda_factor) * inter_cluster_loss
    # return loss

def clustering_quality_loss(embeddings, labels, lambda_value=0.5, margin=1.0):
    distance_matrix = torch.cdist(embeddings, embeddings, p=2)
    # Intra-cluster Loss
    intra_cluster_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    intra_cluster_distances = distance_matrix * intra_cluster_mask.float()
    intra_cluster_loss = intra_cluster_distances.sum() / intra_cluster_mask.sum()
    # Inter-cluster Loss
    inter_cluster_mask = labels.unsqueeze(0) != labels.unsqueeze(1)
    inter_cluster_distances = distance_matrix * inter_cluster_mask.float()
    inter_cluster_loss = F.relu(margin - inter_cluster_distances).sum() / inter_cluster_mask.sum()

    loss = lambda_value * intra_cluster_loss + (1 - lambda_value) * inter_cluster_loss
    return loss



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    args = get_parse()
    alpha = args.alpha


    # Image
    # # Fruit
    dataset_path_dict = {
        "fruit": ["dataset/fruit/species/", "dataset/fruit/color/"]
    }
    dataset_prompt_dict = {
        "fruit": ["Fruit with a species of", "Fruit with a color of"]
    }
    dataset_gpt_dict = {
        "fruit": ["apples, oranges, bananas, strawberries, grapes, raspberries, blueberries, cherries, pears, plums, peaches, nectarines, pineapple, kiwi, watermelon, cantaloupe, apricots", "red, yellow, green, orange, purple, blue"]
    }
    path_list = dataset_path_dict[args.dataset]
    prompt_list = dataset_prompt_dict[args.dataset]
    gpt_list = dataset_gpt_dict[args.dataset]

    # Text
    batch_size = 50
    text_batch = args.batch_size
    for _i, (_data_path, _prompt, _gpt) in enumerate(zip(path_list, prompt_list, gpt_list)):

        model, preprocess = clip.load('ViT-B/32', device, jit=False)

        funcType = type(model.encode_text)
        model.encode_text_with_learnt_tokens = funcType(encode_text_with_learnt_tokens, model)
        model.encode_reference_word = funcType(reference_word_embedding, model)
        model = model.float()

        # for param in model.parameters():
            # param.requires_grad = False
        # for param in model.visual.parameters():
            # param.requires_grad = True

        torch.cuda.empty_cache()
        print("Data path:", _data_path, _prompt)
        img1 = datasets.ImageFolder(root=_data_path, transform=data_transfrom)
        print(img1.class_to_idx, len(img1.class_to_idx))
        idx_to_class = {}
        for key, value in img1.class_to_idx.items():
            idx_to_class[value] = key
        gpt_candidate = _gpt.split(", ")
        print('candidate', len(gpt_candidate), gpt_candidate)

        label_list = []
        image_prob_list = []
        pred_label_list = []
        image_embedding_list = []
        idx = 0
        this_datetime = time.strftime("%m-%d-%Hh%Mm%Ss")

        input_image_list = []
        for img_path, label in img1.imgs:
            if idx % 1000 == 0:
                print(idx)
            idx += 1
            label_list.append(str(label))
            input_image_list.append(preprocess(Image.open(img_path)))


        prompt = prompt_list[_i] + " *"
        asterix_token = clip.tokenize(["*"]).to(device)[0][1]
        prompt_token = clip.tokenize([prompt] * len(input_image_list)).to(device)
        word_token = clip.tokenize(["*"] * len(input_image_list)).to(device)
        concept_prompt_token = clip.tokenize([_data_path.split('/')[2]] * len(input_image_list)).to(device)

        num_images = len(input_image_list)
        num_batch = num_images / batch_size if num_images % batch_size == 0 else num_images // batch_size + 1
        batch_list = [i for i in range(0, num_images, batch_size)]

        text_batch_list = [i for i in range(0, num_images, text_batch)]
        if batch_list[-1] < num_images:
            batch_list.append(num_images)
        if text_batch_list[-1] < num_images:
            text_batch_list.append(num_images)
        batch_block = []
        for i in range(len(text_batch_list) - 1):
            batch_block.append([text_batch_list[i], text_batch_list[i+1]])

        text_features = []
        concept_token = []
        gpt_pred_label_list = []

        image_inputs = torch.tensor(np.stack(input_image_list))

        # obtain reference word
        gpt_inputs = clip.tokenize([f"{_v}" for _v in gpt_candidate]).to(device)
        with torch.no_grad():
            # reference word embedding
            gpt_embeddings = model.encode_reference_word(gpt_inputs)

        # * embedding
        with torch.no_grad():
            for i in range(len(text_batch_list) - 1):
                _prompt_token = word_token[text_batch_list[i]: text_batch_list[i+1]]
                text_features.extend(model.encode_text(_prompt_token))
        text_features = torch.stack(text_features)

        # initialize p_ij with *
        trainable_estimated_tokens = torch.nn.Embedding.from_pretrained(text_features, freeze=False)  # create learnble tokens

        # Alternating Training Loop
        num_alternations = args.alternations
        phase_I_epochs = 100
        phase_II_epochs = 10
        loss_log = [[], []]

        for alternation in range(num_alternations):
            print(f"Starting alternation {alternation + 1}")
            
            # Phase I
            for param in trainable_estimated_tokens.parameters():
                param.requires_grad = True
            for param in model.visual.proj.parameters():
                param.requires_grad = True

            optimizer = optim.Adagrad([
                            {'params': trainable_estimated_tokens.parameters()},
                            {'params': model.visual.proj, 'lr': 1e-5}], lr=args.lr, weight_decay=args.weight_decay)

            for epoch in range(phase_I_epochs):
                loss_values = []
                for j, _block in enumerate(text_batch_list):
                    # Image embeddings
                    _image_inputs = torch.stack(input_image_list[_block[0]: _block[1]]).to(device)
                    _image_embeddings = model.encode_image(_image_inputs)

                    # Proxy embeddings
                    _trainable_estimated_tokens = trainable_estimated_tokens.weight[_block[0]: _block[1]].to(device)
                    weight_matrix = F.softmax(torch.matmul(_trainable_estimated_tokens, gpt_embeddings.T), dim=1)
                    _proxy_embeddings = torch.matmul(weight_matrix, gpt_embeddings)

                    # Prompt embeddings with proxy word
                    _prompt_token = prompt_token[_block[0]: _block[1]]
                    _clip_embeddings = model.encode_text_with_learnt_tokens(_prompt_token, asterix_token, _proxy_embeddings)
                    _clip_embeddings = F.normalize(_clip_embeddings, dim=-1)

                    # Cosine loss for alignment
                    ones = torch.ones(len(_image_inputs)).to(device)
                    criterion = torch.nn.CosineEmbeddingLoss()
                    loss = criterion(_image_embeddings, _clip_embeddings, ones)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_values.append(loss.item())
                if epoch % 10 == 0:
                    avg_loss = np.mean(loss_values)
                    print(f'Phase I Epoch {epoch}\tLoss: {avg_loss:.4f}')
                    loss_log[0].append(epoch)
                    loss_log[1].append(avg_loss)

            # Phase II
            for param in trainable_estimated_tokens.parameters():
                param.requires_grad = False
            for param in model.visual.proj.parameters():
                param.requires_grad = True
            optimizer2 = optim.Adagrad([
                {'params': model.visual.proj, 'lr': 1e-4}], lr=args.lr, weight_decay=args.weight_decay)

            for epoch in range(phase_II_epochs):
                combined_embeddings, text_embedding_list, word_embedding_list, word_eod_embedding_list, image_embedding_list = [], [], [], [], []

                for j in range(len(text_batch_list) - 1):
                    _trainable_estimated_tokens = trainable_estimated_tokens.weight[text_batch_list[j]: text_batch_list[j + 1]].to(device)
                    _trainable_estimated_tokens = torch.mm(_trainable_estimated_tokens, gpt_embeddings.transpose(0, 1))
                    _trainable_estimated_tokens = torch.mm(F.softmax(_trainable_estimated_tokens, dim=1), gpt_embeddings)
                    word_embedding_list.extend(_trainable_estimated_tokens.cpu().detach().numpy())

                    _prompt_token = prompt_token[text_batch_list[j]: text_batch_list[j + 1]]
                    _clip_embeddings = model.encode_text_with_learnt_tokens(_prompt_token, asterix_token, _trainable_estimated_tokens)
                    _clip_embeddings = F.normalize(_clip_embeddings, dim=-1)
                    text_embedding_list.extend(_clip_embeddings.cpu().detach().numpy())

                    with torch.no_grad():
                        _image_inputs = torch.stack(input_image_list[text_batch_list[j]: text_batch_list[j + 1]]).to(device)
                        _image_embeddings = model.encode_image(_image_inputs)
                        image_embedding_list.extend(_image_embeddings.cpu().detach().numpy())

                prompt_embeddings = np.stack(text_embedding_list)
                word_embeddings = np.stack(word_embedding_list)
                image_embeddings = np.stack(image_embedding_list)
                combined_embeddings = np.hstack((word_embeddings, image_embeddings))

                similarity_matrix = torch.matmul(torch.tensor(combined_embeddings).to(device), gpt_embeddings.T)
                pseudo_labels = similarity_matrix.argmax(dim=1).to(device)
                
                clustering_loss = clustering_quality_loss(torch.tensor(combined_embeddings).to(device), pseudo_labels)
                optimizer2.zero_grad()
                clustering_loss.backward()
                optimizer2.step()
                print(f"Phase II Epoch {epoch + 1}: Clustering Quality Loss: {clustering_loss.item()}")
            
            print(f"Alternation {alternation + 1} completed.")