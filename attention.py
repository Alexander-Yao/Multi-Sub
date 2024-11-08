import pickle

from torchvision import transforms, datasets
import torch
import torch.optim as optim
import torch.nn.functional as F
import clip
from PIL import Image
from sklearn.metrics import accuracy_score, normalized_mutual_info_score as nmi, rand_score as ri, adjusted_rand_score as ar
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from parse import get_parse
import time


data_transfrom = transforms.Compose([
    transforms.ToTensor()
])


num_tokens = 77

def encode_text_with_learnt_tokens(self, text, asterix_token, learnable_codes):
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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    args = get_parse()
    alpha = args.alpha
    model, preprocess = clip.load('ViT-B/32', device, jit=False)

    funcType = type(model.encode_text)
    model.encode_text_with_learnt_tokens = funcType(encode_text_with_learnt_tokens, model)
    model = model.float()

    # Image
    # # Fruit
    dataset_path_dict = {
        "fruit": ["dataset/fruit/species/", "dataset/fruit/color/"],
        "fruit360": ["dataset/fruit360/species/", "dataset/fruit360/color/"],
        "cards": ["dataset/cards/number/", "dataset/cards/suits/"],
        "CMUface": ["dataset/CMUface/identity/", "dataset/CMUface/emotion/", "dataset/CMUface/glass/", "dataset/CMUface/pose/"]
    }
    dataset_prompt_dict = {
        "fruit": ["Fruit with a species of", "Fruit with a color of"],
        "fruit360": ["Fruit with a species of", "Fruit with a color of"],
        "cards": ["Card with number of", "Card with suits of"],
        "CMUface": ["Face with identity of", "Face with emotion of", "Face with glass of", "Face with pose of"]
    }
    dataset_gpt_dict = {
        "fruit": ["apples, oranges, bananas, strawberries, grapes, raspberries, blueberries, cherries, pears, plums, peaches, nectarines, pineapple, kiwi, watermelon, cantaloupe, apricots", "red, yellow, green, orange, purple, blue"],
        "fruit360": ["apples, oranges, bananas, strawberries, grapes, raspberries, blueberries, cherries, pears, plums, peaches, nectarines, pineapple, kiwi, watermelon, cantaloupe, apricots", "red, yellow, green, orange, purple, blue"],
        "cards": ["Card with number of", "Card with suits of"],
        "CMUface": ["1, 2, 3, 4, 5, 6, 7, 8, 9, 10", "happiness, sadness, anger, surprise, fear, disgust, contempt", "positive, fashionable", "smiling, frowning, neutral expression, laughing, raised eyebrows, squinting, pouting, wide-eyed, nodding, tilting head"]
    }
    path_list = dataset_path_dict[args.dataset]
    prompt_list = dataset_prompt_dict[args.dataset]
    gpt_list = dataset_gpt_dict[args.dataset]

    # Text
    batch_size = 50
    text_batch = args.batch_size
    for _i, (_data_path, _prompt, _gpt) in enumerate(zip(path_list, prompt_list, gpt_list)):
        torch.cuda.empty_cache()
        print("Data path:", _data_path, _prompt)
        img1 = datasets.ImageFolder(root=_data_path, transform=data_transfrom)
        print(img1.class_to_idx, len(img1.class_to_idx))
        idx_to_class = {}
        for key, value in img1.class_to_idx.items():
            idx_to_class[value] = key
        gpt_candidate = _gpt.split(", ")
        print('candidate', gpt_candidate)

        label_list = []
        image_prob_list = []
        pred_label_list = []
        image_embedding_list = []
        idx = 0
        this_datetime = time.strftime("%m-%d-%Hh%Mm%Ss")

        input_image_list = []
        for img_path, label in img1.imgs:
            # print(idx)
            if idx % 1000 == 0:
                print(idx)
            idx += 1
            label_list.append(str(label))
            input_image_list.append(preprocess(Image.open(img_path)))


        prompt = prompt_list[_i] + " *"
        asterix_token = clip.tokenize(["*"]).to(device)[0][1]
        prompt_token = clip.tokenize([prompt] * len(input_image_list)).to(device)
        concept_prompt_token = clip.tokenize([_data_path.split('/')[2]] * len(input_image_list)).to(device)

        # gpt_init_list = [gpt_candidate[i % len(gpt_candidate)] for i in range(len(input_image_list))]
        # gpt_token = clip.tokenize(gpt_init_list).to(device)
        # print('==', gpt_init_list)
        # gpt_inputs = clip.tokenize([f"{prompt_list[_i]} {_v}" for _v in gpt_candidate]).to(device)

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

        gpt_inputs = clip.tokenize([f"{_v}" for _v in gpt_candidate]).to(device)
        with torch.no_grad():
            gpt_embeddings = model.encode_text(gpt_inputs)
        gpt_embeddings = F.normalize(gpt_embeddings, dim=-1)
        # current_idx = 0
        with torch.no_grad():
            for i in range(len(batch_list) - 1):
                _image_inputs = image_inputs[batch_list[i]: batch_list[i+1]].to(device)
                _image_features = model.encode_image(_image_inputs)
                image_embedding_list.extend(_image_features)
                logits_per_image_gpt, logits_per_text_gpt = model(_image_inputs, gpt_inputs)
                gpt_pred_label_list.extend(torch.argmax(logits_per_image_gpt, dim=-1).cpu().detach().numpy())
            image_embeddings = torch.stack(image_embedding_list)
            image_embeddings = F.normalize(image_embeddings, dim=-1)
        torch.cuda.empty_cache()
        # gpt_labels = [gpt_candidate[_].lower() for _ in gpt_pred_label_list]
        # print(set(gpt_labels))
        # with open(f"pkl/{_data_path.split('/')[1]}_{_data_path.split('/')[2]}.pkl", 'wb') as _w:
        #     pickle.dump(gpt_labels, _w)
        #
        # continue
        # with torch.no_grad():
        #     for i in range(len(text_batch_list) - 1):
        #         print('initial gpt text', i, text_batch_list[i+1])
        #         _prompt_token_list = []
        #         for _gpt_label in gpt_pred_label_list[text_batch_list[i]: text_batch_list[i+1]]:
        #             _prompt_token_list.append(gpt_candidate[_gpt_label])
        #         _prompt_label_token = clip.tokenize(_prompt_token_list).to(device)
        #         text_features.extend(model.encode_text(_prompt_label_token))
        #         _concept_token = concept_prompt_token[text_batch_list[i]: text_batch_list[i+1]]
        #         concept_token.extend(model.encode_text(_concept_token))
        #
        # text_features = torch.stack(text_features)
        # gpt_features = text_features.clone()
        # concept_features = torch.stack(concept_token)
        # concept_prompt_token = concept_prompt_token.cpu()

        with torch.no_grad():
            for i in range(len(text_batch_list) - 1):
                print('initial text', i, text_batch_list[i+1])
                _prompt_token = prompt_token[text_batch_list[i]: text_batch_list[i+1]]
                text_features.extend(model.encode_text(_prompt_token))
                # _concept_token = label_prompt_token[text_batch_list[i]: text_batch_list[i+1]]
                # concept_token.extend(model.encode_text(_concept_token))
        text_features = torch.stack(text_features)
        # concept_features = torch.stack(concept_token)
        # label_prompt_token = label_prompt_token.cpu()

        trainable_estimated_tokens = torch.nn.Embedding.from_pretrained(text_features, freeze=False)  # create learnble tokens

        optimizer = optim.Adagrad(trainable_estimated_tokens.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        ones = torch.ones(len(input_image_list)).to(torch.float16).to(device)
        criterion = torch.nn.CosineEmbeddingLoss()
        criterion2 = torch.nn.MSELoss()
        criterion3 = torch.nn.MSELoss()


        loss_log = [[], []]
        clip_embeddings = []

        for i in range(args.epoch):
            embedding_list = []
            loss = []
            random.shuffle(batch_block)
            for j, _block in enumerate(batch_block):
                _trainable_estimated_tokens = trainable_estimated_tokens.weight[_block[0]: _block[1]].to(device)
                # print('test=', _trainable_estimated_tokens.shape, gpt_embeddings.shape)
                _trainable_estimated_tokens = torch.mm(_trainable_estimated_tokens, gpt_embeddings.transpose(0, 1))
                # print('test', _trainable_estimated_tokens)
                _trainable_estimated_tokens = torch.mm(F.softmax(_trainable_estimated_tokens, dim=1), gpt_embeddings)
                _prompt_token = prompt_token[_block[0]: _block[1]]
                _clip_embeddings = model.encode_text_with_learnt_tokens(_prompt_token, asterix_token, _trainable_estimated_tokens)
                _clip_embeddings = F.normalize(_clip_embeddings, dim=-1)
                _image_embeddings = image_embeddings[_block[0]: _block[1]]
                _loss = criterion(_image_embeddings, _clip_embeddings, ones[_block[0]: _block[1]])
                # _loss = _loss + alpha * criterion2(_trainable_estimated_tokens, concept_features[_block[0]: _block[1]])
                # _loss = _loss + 0.5 * criterion3(_trainable_estimated_tokens, gpt_features[_block[0]: _block[1]])
                loss.append(_loss.cpu().detach().item())

                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()


            if i % 10 == 0:
                # print(gpt_features[0][:5], concept_features[0][:5])
                print(f'Epoch {i}\tLoss: {np.mean(loss):.4}')
                loss_log[0].append(i)
                loss_log[1].append(np.mean(loss))

        # torch.cuda.empty_cache()
        # print(batch_block)
        # for j, _block in enumerate(batch_block):
        #     _trainable_estimated_tokens = trainable_estimated_tokens.weight[_block[0]: _block[1]].to(device)
        #     _prompt_token = prompt_token[_block[0]: _block[1]]
        #     _clip_embeddings = model.encode_text_with_learnt_tokens(_prompt_token, asterix_token,
        #                                                             _trainable_estimated_tokens)


            if  (i + 1) % 100 == 0:
                text_embedding_list = []
                word_embedding_list = []
                for j in range(len(text_batch_list) - 1):
                    _trainable_estimated_tokens = trainable_estimated_tokens.weight[text_batch_list[j]: text_batch_list[j + 1]].to(device)
                    _trainable_estimated_tokens = torch.mm(_trainable_estimated_tokens, gpt_embeddings.transpose(0, 1))
                    _trainable_estimated_tokens = torch.mm(F.softmax(_trainable_estimated_tokens, dim=1), gpt_embeddings)
                    word_embedding_list.extend(_trainable_estimated_tokens.cpu().detach().numpy())
                    _prompt_token = prompt_token[text_batch_list[j]: text_batch_list[j + 1]]
                    _clip_embeddings = model.encode_text_with_learnt_tokens(_prompt_token, asterix_token,
                                                                            _trainable_estimated_tokens)
                    _clip_embeddings = F.normalize(_clip_embeddings, dim=-1)
                    text_embedding_list.extend(_clip_embeddings.cpu().detach().numpy())
                clip_embeddings = np.stack(text_embedding_list)
                word_embeddings = np.stack(word_embedding_list)


                # = clip_embeddings
                # print(loss_log[:][0], loss_log[:][1])

                nmi_list, ar_list, ri_list = [], [], []
                for _i in range(10):
                    kmeans = KMeans(n_clusters=len(img1.class_to_idx), random_state=_i, n_init="auto").fit(text_embedding_list)
                    pred_res = kmeans.labels_
                    nmi_list.append(nmi(label_list, pred_res))
                    ar_list.append(ar(label_list, pred_res))
                    ri_list.append(ri(label_list, pred_res))
                print(f"Text Image: NMI:{np.mean(nmi_list):.4f}\tARI:{np.mean(ar_list):.4f}\tRI:{np.mean(ri_list):.4f}")

                nmi_list2, ar_list2, ri_list2 = [], [], []
                for _i in range(10):
                    kmeans = KMeans(n_clusters=len(img1.class_to_idx), random_state=_i, n_init="auto").fit(word_embedding_list)
                    pred_res = kmeans.labels_
                    nmi_list2.append(nmi(label_list, pred_res))
                    ar_list2.append(ar(label_list, pred_res))
                    ri_list2.append(ri(label_list, pred_res))
                print(f"Word Image: NMI:{np.mean(nmi_list2):.4f}\tARI:{np.mean(ar_list2):.4f}\tRI:{np.mean(ri_list2):.4f}")

                res_path = os.path.join('res/attention', _data_path.split('/')[1])
                if not os.path.exists(res_path):
                    os.makedirs(res_path)
                with open(os.path.join(res_path, 'gpt_' + _data_path.split('/')[2]), 'a') as _a:
                    res_line = f"lr:{args.lr}\twd:{args.weight_decay}\tbatch:{args.batch_size}\t" \
                               f"epoch:{i+1}\talpha:{alpha}\t{this_datetime}\t" \
                               f"prompt:nmi{np.mean(nmi_list):.4f} ari:{np.mean(ar_list):.4f} ri:{np.mean(ri_list):.4f} " \
                               f"word:nmi{np.mean(nmi_list2):.4f} ari:{np.mean(ar_list2):.4f} ri:{np.mean(ri_list2):.4f}\n"
                    _a.write(res_line)
                save_emb = trainable_estimated_tokens.weight.cpu().detach().numpy()
                emb_path = os.path.join('embedding', _data_path.split('/')[1])
                if not os.path.exists(emb_path):
                    os.makedirs(emb_path)
                np.save(os.path.join(emb_path, f"{_data_path.split('/')[2]}_{i+1}"), save_emb)

        # plt.figure(dpi=500)
        # plt.plot(loss_log[0], loss_log[1], markersize=8, color='b')
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        #
        # _tmp_dataset = _data_path.split('/')[1]
        # _tmp_class = _data_path.split('/')[2]
        # save_path = f'plot/loss/{_tmp_dataset}'
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # plt.savefig(os.path.join(save_path, f'loss_{_tmp_class}_{i+1}.png'))
        # plt.show()

