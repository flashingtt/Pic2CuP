# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from functools import partial
from torch.cuda.amp import autocast
import torch.distributed as dist
from tqdm import tqdm
from torchvision.utils import save_image
import sys
import pdb
import logging
import torch.nn.functional as F
from third_party.open_clip.clip import tokenize, _transform
import pickle

from utils import is_master

def prepare_img(img_file, transform):
    return transform(Image.open(img_file))

def visualize_results(model, img2text, args, prompt, dataloader, prompt_learner=None):        
    model.eval()
    img2text.eval()
    if args.prompt == 'learnable':
        prompt_learner.eval()   
    if not os.path.exists(args.demo_out):
        os.makedirs(args.demo_out)        
    if not os.path.exists(os.path.join(args.demo_out, "images")):
        os.makedirs(os.path.join(args.demo_out, "images"))
    text = []
    id_split = tokenize(["*"])[0][1]
    if args.prompt == 'fixed':
        for p in prompt:
            text_tokens = tokenize(p)
            text.append(text_tokens)
            assert id_split in text_tokens
    elif args.prompt == 'learnable':
        ppp = " ".join(["X"] * args.n_ctx)
        img_token = " ".join(["Y"] * args.n_img)
        ttt = ppp + " " + img_token
        for p in prompt:
            text_with_blank = '{}, {}'.format(ttt, p)
            text_tokens = tokenize(text_with_blank)
            text.append(text_tokens)
    text = torch.cat(text, dim=0)    
    text = text.cuda(args.gpu, non_blocking=True)    
    all_image_features, all_image_filenames = [], []
    m = model.module if args.distributed or args.dp else model
    if args.prompt == 'learnable':
        pl = prompt_learner.module if args.distributed or args.dp else prompt_learner
    query_file = args.query_file
    path_save = os.path.join("./data", args.retrieval_data.split('/')[-1].split('.')[0]+".pkl")
    if os.path.exists(path_save):
        with open(path_save, 'rb') as f:
            data = pickle.load(f)
        all_image_features = data['feats']
        all_image_filenames = data['path']
        all_image_features = torch.from_numpy(all_image_features).cuda(args.gpu, non_blocking=True)
    else:
        ## Extract features of target images. 
        with torch.no_grad():
            for batch in tqdm(dataloader):
                images, filenames = batch
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                image_features = m.encode_image(images)           
                image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
                all_image_features.append(image_features)
                for name in filenames:
                    all_image_filenames.append(name)
            all_image_features = torch.cat(all_image_features, dim=0)
            dict_save = {}
            dict_save['feats'] = all_image_features.data.cpu().numpy()
            dict_save['path'] = all_image_filenames
            with open(path_save,"wb") as f:
                pickle.dump(dict_save,f)
    f = open(os.path.join(args.demo_out, "index.html"), 'w')
    html_txt = """"""
    ## For each domain, compute composed features and evaluate.
    for query in query_file.split(","):        
        logging.info("retrieve image of {}".format(query))
        transform = _transform(model.visual.input_resolution)
        query_img = prepare_img(query, transform)
        query_img = torch.unsqueeze(query_img, 0)    
        query_img = query_img.cuda(args.gpu, non_blocking=True)
        img_feature = m.encode_image(query_img) 
        query_img_feature = img2text(img_feature)
        if args.prompt == 'fixed':
            composed_feature = m.encode_text_img_vis(text, query_img_feature, split_ind=id_split)
        elif args.prompt == 'learnable':
            text_embedding = pl(query_img_feature)
            composed_feature = m.encode_text_img_retrieval_learnable(args.n_ctx, args.n_img, text_embedding, text)
        composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)
        img_feature = img_feature / img_feature.norm(dim=-1, keepdim=True)
        text_feature = m.encode_text(text)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        similarity = composed_feature @ all_image_features.T
        _, indices = torch.sort(similarity, descending=True)        
        logging.info("Composed feature result")
        for i, caption in enumerate(prompt):
            logging.info("for prompt {}".format(caption))
            for j, ind in enumerate(indices[i][:8]):
                logging.info("top {} filename {}".format(j, all_image_filenames[ind]))
        image_paths = [[all_image_filenames[ind] for j, ind in enumerate(indices[i][:8])] 
                        for i, caption in enumerate(prompt)]
        html_txt += make_html(prompt, query, image_paths, args.demo_out)
    f.write(html_txt)

def make_html(prompts, query_image, images, path_html):
    import shutil
    html_all = """"""        
    for i in range(len(prompts)):
        prompt = prompts[i]            
        query_image_local = os.path.join(path_html, "images", query_image.split("/")[-1])
        query_image_local_path = os.path.join("images", query_image.split("/")[-1])
        shutil.copy(query_image, query_image_local)
        image_list = images[i]        
        html = """<table><tr>"""    
        html += """<td><p style="display:inline-block;vertical-align;font-size:20px">%s</p></td>"""%(prompt)
        html += """<td><p style="margin-right: 50px;"><img src="%s" height="100"></p></td>"""%(query_image_local_path)
        for image in image_list:
            image_local = os.path.join(path_html, "images", image.split("/")[-1])
            image_path = os.path.join("images", image.split("/")[-1])
            img = os.path.join(os.path.dirname(query_image), image)
            shutil.copy(img, image_local)
            # shutil.copy(image, image_local)
            html += """<td><img src="%s" height=%s></td>"""%(image_path, 200)
        html += """</tr></table>"""
        html_all += html
    return html_all
    #f.write(html_all)


def evaluate_imgnet_retrieval(model, img2text, args, prompt, query_loader, target_loader, prompt_learner=None):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()
    all_image_features = []  
    all_target_labels = []
    all_tar_imagepath = []      
    m = model.module if args.distributed or args.dp else model
    if args.prompt == 'learnable':
        pl = prompt_learner.module if args.distributed or args.dp else prompt_learner
    n_class = 1000
   
    with torch.no_grad():
        ## Extract target image features. 
        for batch in tqdm(target_loader):
            images, labels, tar_img_path = batch
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                labels = labels.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            all_image_features.append(image_features)
            all_target_labels.append(labels)
            all_tar_imagepath.extend(tar_img_path)
            logit_scale = m.logit_scale.exp()
            logit_scale = logit_scale.mean()   

        ## Extract query features 
        recall_result = []
        for p_ind, p in enumerate(prompt):
            ## which token has to be replaced with image features
            id_split = tokenize(["*"])[0][1]
            text = tokenize(p).view(1, -1)
            text = text.cuda(args.gpu, non_blocking=True)
            ## text only features (domain name only)
            if args.prompt == 'fixed':
                text_only = p.replace("*", "")
            elif args.prompt == 'learnable':
                text_only = p.split(",")[1]
                print(text_only)

            text_only = tokenize(text_only).view(1, -1)            
            text_only = text_only.cuda(args.gpu, non_blocking=True)                        
            text_only_features = m.encode_text(text_only)
            text_only_features_normed = text_only_features / text_only_features.norm(dim=-1, keepdim=True)

            all_query_features = []
            all_query_image_features = []
            all_query_mixture_features = []
            all_query_labels = []
            all_text_features = []
            all_query_imgpath = []
            for batch in tqdm(query_loader):
                images, labels, query_imgpath = batch
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                    labels = labels.cuda(args.gpu, non_blocking=True)
                ## Label is decided by class label and images' domain
                labels += n_class * p_ind
                image_features = m.encode_image(images)
                 ## Composed feature extraction
                image_features_query = img2text(image_features)
                if args.prompt == 'fixed':
                    if args.f == 'im2text':
                        composed_feature = m.encode_text_img_retrieval(text, image_features_query, split_ind=id_split)                            
                    elif args.f == 'im2multitext':
                        raise NotImplementedError
                elif args.prompt == 'learnable':
                    text_embedding = pl(image_features_query)
                    composed_feature = m.encode_text_img_retrieval_learnable(args.n_ctx, args.n_img, text_embedding, text)
                composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)            
                ## Image feature only
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)  
                ## average of image and text features
                mixture_features = image_features + text_only_features_normed
                mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)       

                all_text_features.append(text_only_features_normed.repeat((image_features.shape[0], 1)))
                all_query_features.append(composed_feature)
                all_query_image_features.append(image_features)
                all_query_mixture_features.append(mixture_features)
                all_query_labels.append(labels)
                all_query_imgpath.extend(query_imgpath)

            metric_func = partial(get_metrics_imgnet,
                query_imgpath=all_query_imgpath, 
                tar_imgpath=all_tar_imagepath,
                image_features=torch.cat(all_image_features), 
                query_labels=torch.cat(all_query_labels),
                target_labels=torch.cat(all_target_labels),
                )

            feats = {'composed': torch.cat(all_query_features), 
                    'image': torch.cat(all_query_image_features),
                    'text': torch.cat(all_text_features),
                    'mixture': torch.cat(all_query_mixture_features)}        

            for key, value in feats.items():
                metrics = metric_func(args, key, query_features=value)
                logging.info(
                f"Eval {key} Feature"
                + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
                print(f"Eval {key} Feature " + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
                if args.val_result_txt is not None:
                    if key == 'composed':
                        style = p.split(",")[1].split(" ")[2]
                        line = f"{style}\tR@10: {metrics['Real2Sketch_R@10']:.4f}\tR@50: {metrics['Real2Sketch_R@50']:.4f}"
                        recall_result.append(line)
        if args.val_result_txt is not None:
            with open(args.val_result_txt, 'a') as f:
                l = "\t".join(recall_result)
                f.write(l + "\n")

    return metrics


def evaluate_coco(model, img2text, args, loader, prompt_learner=None):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()
    if args.prompt == 'learnable':
        prompt_learner.eval()

    all_image_features = []  
    all_query_image_features = []  
    all_mixture_features = []  
    all_composed_features_with_class = []  
    all_text_full_features = []
    all_file_names = []

    m = model.module if args.distributed or args.dp else model
    if args.prompt == 'learnable':
        p = prompt_learner.module if args.distributed or args.dp else prompt_learner
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean()
    with torch.no_grad():
        for batch in tqdm(loader):
            images, region_images, text_full, text_with_blank, text_with_blank_query, filename, raw_text = batch            
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                region_images = region_images.cuda(args.gpu, non_blocking=True)
                text_full = text_full.cuda(args.gpu, non_blocking=True)
                text_with_blank = text_with_blank.cuda(args.gpu, non_blocking=True)
                text_with_blank_query = text_with_blank_query.cuda(args.gpu, non_blocking=True)

            

            ## Target image features 
            image_features = m.encode_image(images)             
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  
            id_split = tokenize(["*"])[0][1]
            ## Composed image features
            query_image_features = m.encode_image(region_images)
            query_image_tokens = img2text(query_image_features)
            if args.prompt == 'fixed':
                if args.f == 'im2text':          
                    composed_feature_with_class = m.encode_text_img_retrieval(text_with_blank_query, query_image_tokens, split_ind=id_split, repeat=False)                        
                elif args.f == 'im2multitext':
                    composed_feature_with_class = m.encode_text_img_retrieval(text_with_blank_query, query_image_tokens, split_ind=id_split, repeat=False)
            elif args.prompt == 'learnable':
                text_embedding = p(query_image_tokens)
                composed_feature_with_class = m.encode_text_img_retrieval_learnable(args.n_ctx, args.n_img, text_embedding, text_with_blank_query)
            
            composed_feature_with_class = composed_feature_with_class / composed_feature_with_class.norm(dim=-1, keepdim=True)        
            ## Text only features
            text_full_features = m.encode_text(text_full)
            text_full_features = text_full_features / text_full_features.norm(dim=-1, keepdim=True)            
            ## Query only features
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)                               
            ## Mixed featurs
            mixture_features = query_image_features + text_full_features
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)            

            all_image_features.append(image_features.cpu())
            all_text_full_features.append(text_full_features.cpu())       
            all_query_image_features.append(query_image_features.cpu())
            all_mixture_features.append(mixture_features.cpu())                        
            all_composed_features_with_class.append(composed_feature_with_class.cpu())            
            all_file_names.extend(filename)

        metric_func = partial(get_metrics_coco, 
                file_names = all_file_names,
                image_features=torch.cat(all_image_features), 
                logit_scale=logit_scale
                )
        feats = {'composed': torch.cat(all_composed_features_with_class), 
                 'image': torch.cat(all_query_image_features),
                 'text': torch.cat(all_text_full_features),
                 'mixture': torch.cat(all_mixture_features)}        

        for key, value in feats.items():
            metrics = metric_func(args, key, ref_features=value)
            logging.info(
            f"Eval {key} Feature"
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
            print(f"\nEval {key} Feature" + "\t" + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
            if args.val_result_txt is not None:
                if key == 'composed':
                    avg_ft_recall = (metrics['ref_to_image_R@1'] + metrics['ref_to_image_R@5'] + metrics['ref_to_image_R@10']) / 3
                    avg_recall = (metrics['ref_to_image_R@10'] + metrics['ref_to_image_R@50']) / 2
                    total_avg_r = (metrics['ref_to_image_R@1'] + metrics['ref_to_image_R@5'] + metrics['ref_to_image_R@10'] + metrics['ref_to_image_R@50'] + metrics['ref_to_image_R@100']) / 5
                    with open(args.val_result_txt, 'a') as f:
                        line = f"R@1: {metrics['ref_to_image_R@1']:.5f}" + "\t" + \
                            f"R@5: {metrics['ref_to_image_R@5']:.5f}" + "\t" + \
                            f"R@10: {metrics['ref_to_image_R@10']:.5f}" + "\t" + \
                            f"R@50: {metrics['ref_to_image_R@50']:.5f}" + "\t" + \
                            f"R@100: {metrics['ref_to_image_R@100']:.5f}" + "\t" + \
                            f"ft_Avg_R: {avg_ft_recall:.5f}" + "\t" + f"Avg_R: {avg_recall:.5f}" + "\t" + f"Total_Avg_R: {total_avg_r:.5f}" + "\t" + \
                            f"{metrics['image_to_ref_R@1']:.5f}" + "\t" + f"{metrics['image_to_ref_R@5']:.5f}" + "\t" + f"{metrics['image_to_ref_R@10']:.5f}" + "\t" + f"{metrics['image_to_ref_R@50']:.5f}" + "\t" + f"{metrics['image_to_ref_R@100']:.5f}"
                            
                        f.write(line + "\n")

    return metrics


def evaluate_cirr(model, img2text, args, query_loader, target_loader, prompt_learner=None):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()
    if args.prompt == 'learnable':
        prompt_learner.eval()
    all_image_features = []  
    all_query_image_features = []  
    all_composed_features = []  
    all_mixture_features = []  
    all_caption_features = []  
    all_ref_paths = []
    all_target_paths = []
    all_answer_paths = []
    all_raw_captions = []
    all_group_members = []
    m = model.module if args.distributed or args.dp else model
    if args.prompt == 'learnable':
        p = prompt_learner.module if args.distributed or args.dp else prompt_learner
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean()   

    with torch.no_grad():
        for batch in tqdm(target_loader):
            target_images, target_paths = batch
            if args.gpu is not None:
                target_images = target_images.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)

        for batch in tqdm(query_loader):
            ref_images, text_with_blank, caption_only, ref_paths, answer_paths, raw_captions, group_members = batch
            if args.gpu is not None:
                ref_images = ref_images.cuda(args.gpu, non_blocking=True)
                text_with_blank = text_with_blank.cuda(args.gpu, non_blocking=True)
                caption_only = caption_only.cuda(args.gpu, non_blocking=True)
            id_split = tokenize(["*"])[0][1]                        
            for path in ref_paths:
                all_ref_paths.append(path)
            for path in answer_paths:
                all_answer_paths.append(path)
            for cap in raw_captions:
                all_raw_captions.append(cap)
            # print(ref_paths)
            group_members = np.array(group_members).T.tolist() # **************************************!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # print(group_members)
            # print(len(group_members))
            for gpmbs in group_members:
                all_group_members.append(gpmbs)

            caption_features = m.encode_text(caption_only)
            ## Composed features
            query_image_features = m.encode_image(ref_images)
            query_image_tokens = img2text(query_image_features)
            if args.prompt == 'fixed':
                if args.f == 'im2text':
                    composed_feature = m.encode_text_img_retrieval(text_with_blank, query_image_tokens, split_ind=id_split, repeat=False)
                elif args.f == 'im2multitext':
                    composed_feature = m.encode_text_img_retrieval_multi(text_with_blank, query_image_tokens, split_ind=id_split, repeat=False)
            elif args.prompt == 'learnable':
                text_embedding = p(query_image_tokens)
                composed_feature = m.encode_text_img_retrieval_learnable(args.n_ctx, args.n_img, text_embedding, text_with_blank)                

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)                       
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)   
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)            
            mixture_features = query_image_features + caption_features            
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)
            all_caption_features.append(caption_features)
            all_query_image_features.append(query_image_features)
            all_composed_features.append(composed_feature)            
            all_mixture_features.append(mixture_features)                        

        all_target_paths = np.array(all_target_paths)
        all_ref_paths = np.array(all_ref_paths)
        all_answer_paths = np.array(all_answer_paths)
        
        metric_func = partial(get_metrics_cirr, 
                image_features=torch.cat(all_image_features), 
                reference_names=all_ref_paths, 
                index_names=all_target_paths, 
                target_names=all_answer_paths,
                group_members=all_group_members)

        feats = {'composed': torch.cat(all_composed_features), 
                 'image': torch.cat(all_query_image_features),
                 'text': torch.cat(all_caption_features),
                 'mixture': torch.cat(all_mixture_features)}
        
        for key, value in feats.items():
            metrics = metric_func(args, key, ref_features=value)
            logging.info(
            f"Eval {key} Feature"
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
            print(f"\nEval {key} Feature" + "\t" + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
            if args.val_result_txt is not None:
                if key == 'composed':
                    avg_recall = (metrics['recall_R@10'] + metrics['recall_R@50']) / 2
                    avg_recall_s = (metrics['recall_R@5'] + metrics['R_subset@1']) / 2
                    total_recall = (metrics['recall_R@1'] + metrics['recall_R@5'] + metrics['recall_R@10'] + metrics['recall_R@50'] + metrics['recall_R@100']) / 5
                    with open(args.val_result_txt, 'a') as f:
                        line = "\t".join(f"{k}: {v:.4f}" for k, v in metrics.items()) + "\t" + \
                                f"Avg_R: {avg_recall_s:.4f}" + "\t" + \
                                f"Avg_10_50: {avg_recall:.4f}" + "\t" + \
                                f"total_avg: {total_recall:.4f}"
                        f.write(line + "\n")

    return metrics


def evaluate_cirr_test(model, img2text, args, query_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()

    all_image_features = []  
    all_query_image_features = []  
    all_composed_features = []  
    all_composed_plus_image_features = []  
    all_mixture_features = []  
    all_caption_features = []  
    all_ref_paths = []
    all_target_paths = []
    all_answer_paths = []
    all_ids = []

    m = model.module if args.distributed or args.dp else model   
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean()   

    with torch.no_grad():
        for batch in tqdm(target_loader):
            target_images, target_paths = batch
            if args.gpu is not None:
                target_images = target_images.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)

        for batch in tqdm(query_loader):
            ref_images, text_with_blank, caption_only, ref_paths, pairids = batch
            if args.gpu is not None:
                ref_images = ref_images.cuda(args.gpu, non_blocking=True)
                text_with_blank = text_with_blank.cuda(args.gpu, non_blocking=True)
                caption_only = caption_only.cuda(args.gpu, non_blocking=True)
            id_split = tokenize(["*"])[0][1]                        
            for ids in pairids:
                all_ids.append(ids)
            for path in ref_paths:
                all_ref_paths.append(path)

            caption_features = m.encode_text(caption_only)
            query_image_features = m.encode_image(ref_images)

            if args.eval_combiner:
                composed_feature = img2text(query_image_features, caption_features)
            else:
                query_image_tokens = img2text(query_image_features)
                composed_feature = m.encode_text_img_retrieval(text_with_blank, query_image_tokens, split_ind=id_split, repeat=False)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)                       
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)   
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)            
            mixture_features = query_image_features + caption_features
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)
            all_caption_features.append(caption_features)
            all_query_image_features.append(query_image_features)
            all_composed_features.append(composed_feature)            
            all_mixture_features.append(mixture_features)            

        all_target_paths = np.array(all_target_paths)
        all_ref_paths = np.array(all_ref_paths)
        all_answer_paths = np.array(all_answer_paths)
        res_all = {}
        metrics_func = partial(get_cirr_testoutput, 
                               image_features=torch.cat(all_image_features),
                               reference_names=all_ref_paths,
                               index_names=all_target_paths,
                               id_names=all_ids)
        feats = {'composed': torch.cat(all_composed_features), 
                 'image': torch.cat(all_query_image_features),
                 'text': torch.cat(all_caption_features),
                 'mixture': torch.cat(all_mixture_features)}        
        for key, value in feats:
            res_all[key] = metrics_func(ref_features=value)
    return res_all


def evaluate_fashion(model, img2text, args, source_loader, target_loader, prompt_learner=None):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()
    if args.prompt == 'learnable':
        prompt_learner.eval()
    all_target_paths = []
    all_answer_paths = []
    all_image_features = []  
    all_query_image_features = []  
    all_composed_features = []  
    all_caption_features = []  
    all_mixture_features = []  
    all_reference_names = []
    all_captions = []     
    m = model.module if args.distributed or args.dp else model
    if args.prompt == 'learnable':
        p = prompt_learner.module if args.distributed or args.dp else prompt_learner
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean() 

    with torch.no_grad():
        for batch in tqdm(target_loader):
            target_images, target_paths = batch
            if args.gpu is not None:
                target_images = target_images.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)

    with torch.no_grad():
        for batch in tqdm(source_loader):
            ref_images, target_images, target_caption, modified_only, answer_paths, ref_names, captions = batch
            for path in answer_paths:
                all_answer_paths.append(path)
            all_reference_names.extend(ref_names)
            all_captions.extend(captions)
            if args.gpu is not None:
                ref_images = ref_images.cuda(args.gpu, non_blocking=True)
                target_images = target_images.cuda(args.gpu, non_blocking=True)
                target_caption = target_caption.cuda(args.gpu, non_blocking=True)
                modified_only = modified_only.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            query_image_features = m.encode_image(ref_images)
            id_split = tokenize(["*"])[0][1]            
            caption_features = m.encode_text(target_caption)                            
            query_image_tokens = img2text(query_image_features)
            if args.prompt == 'fixed':
                caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)
                if args.f == 'im2text':
                    composed_feature = m.encode_text_img_retrieval(target_caption, query_image_tokens, split_ind=id_split, repeat=False)
                elif args.f == 'im2multitext':
                    composed_feature = m.encode_text_img_retrieval_multi(target_caption, query_image_tokens, split_ind=id_split, repeat=False)
            elif args.prompt == 'learnable':
                text_embedding = p(query_image_tokens)
                composed_feature = m.encode_text_img_retrieval_learnable(args.n_ctx, args.n_img, text_embedding, target_caption)
                modified_features = m.encode_text(modified_only)
                caption_features = modified_features / modified_features.norm(dim=-1, keepdim=True)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            # caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)                       
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)   
            mixture_features = query_image_features + caption_features
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)

            all_caption_features.append(caption_features)
            all_query_image_features.append(query_image_features)
            all_composed_features.append(composed_feature)            
            all_mixture_features.append(mixture_features)   

        # image_features represent the image features in database
        # target_names represent the image path in database
        # answer_names represent the target image path in triplet
        metric_func = partial(get_metrics_fashion, 
                              ref_names=all_reference_names,
                              image_features=torch.cat(all_image_features),
                              target_names=all_target_paths, answer_names=all_answer_paths)
        feats = {'composed': torch.cat(all_composed_features), 
                 'image': torch.cat(all_query_image_features),
                 'text': torch.cat(all_caption_features),
                 'mixture': torch.cat(all_mixture_features)}
        
        for key, value in feats.items():
            metrics = metric_func(args, key, ref_features=value)
            logging.info(
            f"Eval {key} Feature"
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
            print(f"\nEval {key} Feature" + "\t" + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
            if args.val_result_txt is not None:
                if key == 'composed':
                    avg_recall = (metrics['R@10'] + metrics['R@50']) / 2
                    total_recall = (metrics['R@1'] + metrics['R@5'] + metrics['R@10'] + metrics['R@50'] + metrics['R@100']) / 5
                    with open(args.val_result_txt, 'a') as f:
                        line = '\t'.join(f"{k}: {v:.4f}" for k, v in metrics.items()) + "\t" + f"Avg_R: {avg_recall:.4f}" + "\t" + f"Total_Avg_R: {total_recall:.4f}"
                        f.write(line + "\n")

    return metrics


def get_metrics_coco(args, key, file_names, image_features, ref_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale.cpu() * image_features @ ref_features.t()).detach().cpu()
    logits_per_ref = logits_per_image.t().detach().cpu()
    logits = {"image_to_ref": logits_per_image, "ref_to_image": logits_per_ref}
    ground_truth = torch.arange(len(ref_features)).view(-1, 1)
    
    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        
        preds = torch.where(ranking == ground_truth)[1]
        # visualization
        if key == 'composed' and name == 'ref_to_image' and args.visual_best:
            r_at1 = torch.where(preds == 0)[0].detach().cpu().numpy()
            for i in range(len(r_at1)):
                final_line = []
                line = np.array(file_names)[ranking[r_at1[i]][:5]]
                for l in line:
                    final_line.append(os.path.splitext(l)[0])
                with open('./visual_result/coco.txt', 'a') as f:
                    f.write("\t".join(final_line)+"\n")
        elif args.visual_compared and key == 'composed' and name == 'ref_to_image':
            for i in range(ranking.shape[0]):
                final_line = []
                line = np.array(file_names)[ranking[i][:5]]
                for l in line:
                    final_line.append(os.path.splitext(l)[0])
                with open('./visual_compared_result/coco.txt', 'a') as f:
                    f.write("\t".join(final_line)+"\n")
            
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10, 50, 100]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)
            
    return metrics


def get_metrics_fashion(args, key, ref_names, image_features, ref_features, target_names, answer_names):
    """
    image_features: (6346, 768) shirt image in database
    ref_features: (2038, 768)
    target_names: image path in database
    answer_names: target path in triplet
    """
    metrics = {}
    distances = 1 - ref_features @ image_features.T     # the distance is more closer to 1, the better (2038, 6346)
    sorted_indices = torch.argsort(distances, dim=-1).cpu()     # from small to large for each line
    sorted_index_names = np.array(target_names)[sorted_indices]
    
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(answer_names), len(target_names)).reshape(len(answer_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(answer_names)).int())
    # Compute the metrics
    for k in [1, 5, 10, 50, 100]:
        metrics[f"R@{k}"] = (torch.sum(labels[:, :k]) / len(labels)).item() * 100
        # metrics[f"R@{k}"] = (torch.sum(labels[:, -k:]) / len(labels)).item() * 100
    # visulization
    if args.visual_best and key == 'composed':
        txt_name = './visual_result/fashioniq_' + args.source_data + '.txt'
        with open(txt_name, 'a') as f:
            for i in range(labels.shape[0]):
                if torch.sum(labels[i][:1]) == 1:
                    line = sorted_index_names[i][:5]
                    final_line = [os.path.splitext(os.path.basename(ref_names[i]))[0]+":"]
                    for l in line:
                        img_name = os.path.splitext(os.path.basename(l))[0]
                        final_line.append(img_name)
                    f.write("\t".join(final_line)+"\n")
    elif args.visual_compared and key == 'composed':
        txt_name = './visual_compared_result/fashioniq_' + args.source_data + '.txt'
        with open(txt_name, 'a') as f:
            for i in range(labels.shape[0]):
                line = sorted_index_names[i][:5]
                final_line = [os.path.splitext(os.path.basename(ref_names[i]))[0]+":"]
                for l in line:
                    img_name = os.path.splitext(os.path.basename(l))[0]
                    final_line.append(img_name)
                f.write("\t".join(final_line)+"\n")
    return metrics


def get_metrics_cirr(args, key, image_features, ref_features, reference_names, index_names, target_names, group_members):
    """
    image_features: (2297, 768)
    ref_features: (4181, 768)
    reference_names: (4181,)
    index_names: (2297,) image path in database
    target_names: (4181,) target path in triplet
    """
    metrics = {}
    distances = 1 - ref_features @ image_features.T 
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), 
        len(index_names)).reshape(len(target_names), -1))
    # pdb.set_trace()        
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)    # 9603757 - 9599576 = 4181
    
    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))
    
    # Compute the subset predictions
    # print(group_members)
    # print(len(group_members))
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    for k in [1, 5, 10, 50, 100]:
        metrics[f"recall_R@{k}"] = (torch.sum(labels[:, :k]) / len(labels)).item() * 100

    if args.visual_best and key == 'composed':
        with open('./visual_result/cirr.txt', 'a') as f:
            for i in range(labels.shape[0]):
                if torch.sum(labels[i][:1]) == 1:
                    line = sorted_index_names[i][:5]
                    final_line = [os.path.splitext(reference_names[i])[0]+":"]
                    for l in line:
                        img_name = os.path.splitext(os.path.basename(l))[0]
                        final_line.append(img_name)
                    f.write("\t".join(final_line)+"\n")
    elif args.visual_compared and key == 'composed':
        with open('./visual_compared_result/cirr.txt', 'a') as f:
            for i in range(labels.shape[0]):
                line = sorted_index_names[i][:5]
                final_line = [os.path.splitext(reference_names[i])[0]+":"]
                for l in line:
                    img_name = os.path.splitext(os.path.basename(l))[0]
                    final_line.append(img_name)
                f.write("\t".join(final_line)+"\n")

    for k in [1, 2, 3]:
        metrics[f"R_subset@{k}"] = (torch.sum(group_labels[:, :k]) / len(group_labels)).item() * 100

    # with open('./visual_result/cirr_subset.txt', 'a') as g:
    #     for i in range(group_labels.shape[0]):
    #         line = sorted_index_names[..., None][i][:5]
    #         final_line = []
    #         for l in line:
    #             print(l)
    #             img_name = os.path.splitext(os.path.basename(l))[0]
    #             final_line.append(img_name)
    #         g.write("\t".join(final_line)+"\n")

    return metrics


def get_cirr_testoutput(image_features, ref_features, reference_names, index_names, id_names):
    metrics = {}
    distances = 1 - ref_features @ image_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    result_dict = {"version": "rc2", "metric": "recall"}
    for ind in range(len(id_names)):
        pairid = str(id_names[ind].item())
        result_dict[pairid] = []
        for t in range(50):
            result_dict[pairid].append(sorted_index_names[ind][t].replace(".png", ""))
    return result_dict


def get_metrics_imgnet(args, key, query_features, query_imgpath, tar_imgpath, image_features, query_labels, target_labels):
    metrics = {}
    num_classes = 7000
    # print(query_labels.shape)
    # print(target_labels.shape)
    # print(query_features.shape) # (10000, 768)
    # print(image_features.shape) # (16983, 768)
    # print(len(query_imgpath)) # (10000)
    # print(len(tar_imagepath)) # (16983)
    query_onehot = F.one_hot(query_labels, num_classes=num_classes).float() # (10000, 7000)
    
    target_onehot = F.one_hot(target_labels, num_classes=num_classes).float() # (16983, 7000)
    batches = [(query_features[x:x+100], query_onehot[x:x+100], query_imgpath[x:x+100]) for x in range(0, len(query_features), 100)]
    # print('11111: ', len(batches)) # (100)
    for k in [1, 5, 10, 50, 100, 200]:
        metrics[f"Real2Sketch_R@{k}"] = 0
        metrics[f"Real2Sketch_P@{k}"] = 0
    for batch in batches:
        # batch ()
        feats, labels, q_imgpath = batch[0], batch[1], batch[2] # (100, 768) (100, 7000) (100)
        logits_per_query = (feats @ image_features.t()).detach().cpu() # (100, 16983)
        label_matrix = (labels @ target_onehot.t()).detach().cpu() # (100, 16983)           
        ranking = torch.argsort(logits_per_query, descending=True) # (100, 16983)
        for k in [1, 5, 10, 50, 100, 200]:
            matrix_k = torch.zeros_like(label_matrix) # (100, 16983)
            rank_k = ranking[:, :k] # (100, k)
            matrix_k[torch.arange(matrix_k.size(0)).unsqueeze(1), rank_k] = 1
            consistency = matrix_k * label_matrix # (100, 16983)
            num_correct = torch.sum(consistency, dim=1)
            # print(num_correct)
            num_predicted = torch.sum(matrix_k, dim=1)            
            num_total = torch.sum(label_matrix, dim=1)
            recall = torch.mean(num_correct / (num_total+1e-5))
            # print(recall)
            precision = torch.mean(num_correct / num_predicted)
            metrics[f"Real2Sketch_R@{k}"] += recall * len(feats)
            metrics[f"Real2Sketch_P@{k}"] += precision * len(feats)
            if key == 'composed' and k == 5 and args.visual_best:
                m1 = torch.zeros_like(label_matrix)
                r1 = ranking[:, :1]
                m1[torch.arange(m1.size(0)).unsqueeze(1), r1] = 1
                c1 = m1 * label_matrix
                for i in range(100):
                    if len(np.array(tar_imgpath)[np.where(c1[i] == 1)]) == 1:
                        r_at5 = np.array(tar_imgpath)[np.where(consistency[i] == 1)]
                        recall_at5 = []
                        for r5 in r_at5:
                            recall_at5.append(os.path.splitext(os.path.basename(r5))[0])
                        final_line = "\t".join(recall_at5)
                        line = f"{os.path.splitext(os.path.basename(q_imgpath[i]))[0]}:\t{final_line}\n"
                        with open('./visual_result/imgnet.txt', 'a') as f:
                            f.write(line)
            elif key == 'composed' and k == 5 and args.visual_compared:
                for i in range(100):
                    r_at5 = np.array(tar_imgpath)[np.where(consistency[i] == 1)]
                    recall_at5 = []
                    for r5 in r_at5:
                        recall_at5.append(os.path.splitext(os.path.basename(r5))[[0]])
                    final_line = "\t".join(recall_at5)
                    line = f"{os.path.splitext(os.path.basename(q_imgpath[i]))[0]}:\t{final_line}\n"
                    with open('./visual_compared_result/imgnet.txt', 'a') as f:
                        f.write(line)

    for k in [1, 5, 10, 50, 100, 200]:
        metrics[f"Real2Sketch_R@{k}"] /= len(query_features)
        metrics[f"Real2Sketch_P@{k}"] /= len(query_features)
    return metrics
