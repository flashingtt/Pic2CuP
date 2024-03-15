from third_party.open_clip.clip import tokenize
from third_party.open_clip import clip
import torch
from argparse import ArgumentParser
import pdb
import os

# device = 'cuda:3'

# word = 'X X X'

# token = tokenize(word).to(device)
# print(token)

# clip_model, _ = clip.load('ViT-L/14', device=device)

# embedding = clip_model.token_embedding(token)
# # pdb.set_trace()
# print(embedding.shape)

# a = torch.tensor([[1, 5, 4, 14, 0, 0, 0, 0, 0, 0],
#                   [1, 3, 6, 7, 1, 14, 0, 0, 0, 0],
#                   [1, 7, 9, 3, 2, 8, 2, 14, 0, 0]])

# b = torch.tensor([[1, 2, 3, 4, 14, 0, 0, 0, 0, 0],
#                   [1, 2, 14, 0, 0, 0, 0, 0, 0, 0],
#                   [1, 2, 3, 4, 5, 6, 7, 8, 14, 0]])

# c = torch.zeros_like(a)
# print(c)

# for i in range(a.shape[0]):
    

#     c[i] = torch.cat([a[:a[i].argmax(dim=-1)], b[1:]], dim=1)
#     if len(c[i]) > a.shape[1]:
#         c[i][-1] = 14


# # c = torch.cat([a[:, :a.argmax(dim=-1)], b[]], dim=1)
# print(c)

# a = torch.tensor([[1,2,3,4],
#                   [5,6,7,8],
#                   [9,10,11,12]])
# b = torch.tensor([[True, True, True, False],
#                   [True, False, True, True],
#                   [False, True, True, True]])
# print(a.shape)
# print(a[b].shape)
# print(a[b].reshape(a.shape[0], a.shape[1]-1))


# path = '/home/jumpserver/yxt/cir/composed_image_retrieval/logs/lr=0.0001_wd={args.wd}_agg={args.aggregate}_model={args.model}_batchsize={args.batch_size}_workers={args.workers}_date=2024-03-04-16-07-24/checkpoints'

# model_dict = {}

# for root, _, filenames in os.walk(path):
#     for filename in filenames:
#         id = os.path.splitext(filename)[0].replace('epoch_', '')
#         model_path = os.path.join(root, filename)
#         # print(id, '\t', model_path)
#         model_dict[int(id)] = model_path


# model_dict = {key: model_dict[key] for key in sorted(model_dict.keys())}
# print(model_dict)

# mes = 'X X X Y, in toy style.'
# print(mes.split(",")[1].split(" ")[2])

import numpy as np
import torch.nn.functional as F
# index = [1, 3, 5, 7, 9]

# nums = [1,2,3,4,5,6,7,8,9,10]

# print(np.array(nums)[index[:3]])

# a = torch.arange(10).unsqueeze(1)
# print(a)
# print(a.shape)
# query_labels = torch.tensor([2,2,2,4,5,6,7,9,1,963,2304])
# print(query_labels.shape)
# query_onehot = F.one_hot(query_labels, num_classes=7000).float()
# print(query_onehot.shape)

# a = torch.rand(10000, 768)
# b = torch.rand(10000, 7000)

# batches = [(a[x:x+100], b[x:x+100]) for x in range(0, len(a), 100)]
# print(batches[0][1].shape)

# logits_per_query = torch.rand(10, 15)

# matrix_k = torch.zeros(10, 15)

# ranking = torch.argsort(logits_per_query, descending=True)

# rank_k = ranking[:, :5]

# print(rank_k)
# c = torch.arange(matrix_k.size(0)).unsqueeze(1)
# print(c.shape)

# matrix_k[torch.arange(matrix_k.size(0)).unsqueeze(1), rank_k] = 1
# print(matrix_k)

# e = torch.rand(100,15)
# f = torch.rand(100, 15)
# g = e * f

# print(g.shape)

# x = []
# if x is not None:
#     print('xxx')
#     print(len(x))

# nums = [0,0,0,1,0,0,1,0,0,1,0]

# index = np.where(np.array(nums) == 1)
# print(index)


# model = torch.load('/home/jumpserver/.cache/clip/ViT-L-14.pt', map_location='cuda:4')

# def count_parameters(model):
#     nums = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return nums

# nums = count_parameters(model)



# print(nums)


from third_party.open_clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

sot_token = _tokenizer.encoder["<|startoftext|>"]
eot_token = _tokenizer.encoder["<|endoftext|>"]

print(sot_token)
print(eot_token)
