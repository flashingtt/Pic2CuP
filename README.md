# Pic2CuP

### Pic2CuP: Mapping Pictures to Customized Prompts

## Getting Started

### Installation

1. Clone the repo

```sh
git clone https://github.com/flashingtt/Pic2CuP.git
```

2. Install environment
```sh
conda create -n pic2cup python=3.8 -y
conda activate pic2cup
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirement.txt
```

## Preparing

### Prepare Datasets

1. Git LFS setup

```sh
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```

2. Download datasets

* Get training dataset CC3M from https://huggingface.co/datasets/yxchng/cc3m_01122022

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/yxchng/cc3m_01122022
cd cc3m_01122022
git lfs pull --include="*.tar"
mkdir cc3m
tar -xvf *.tar -c ./cc3m
```

* Get validation dataset FashionIQ from https://github.com/XiaoxiaoGuo/fashion-iq
* Get validation dataset CIRR from https://github.com/Cuberick-Orion/CIRR
* Get validation dataset COCO from https://cocodataset.org/#download

```sh
wget http://images.cocodataset.org/zips/unlabeled2017.zip
wget http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip
```

Run the python file ```./data/coco/prepared_data.py```, remember to modify the relative path

* Get validation dataset ImageNet from https://github.com/hendrycks/imagenet-r

```sh
wget https://github.com/hendrycks/imagenet-r
```

Run the python file ```./data/imgnet/split_real.py```, remember to modify the relative path

3. Go to https://ai.google.com/research/ConceptualCaptions/download and press download button to get a 500MB .tsv file

* Reconstruct the file

```python
    import os
    import pandas as pd

    data_path = './cc3m'
    save_path = './cc3m'

    save_file_path = os.path.join(save_path, "Train_GCC-training_output.csv") 
    csv_list = []
    for root, dirs, files in tqdm(os.walk(data_path)):
        for file in files:
            if file[-4:] == ".jpg" and file[:2] != "._":
                filepath, txt_file_path = os.path.join(root, file), os.path.join(root, file[:-4]+".txt")
                title = read_txt_file(txt_file_path)
                csv_list.append({"filepath":filepath, "title":title})
                # print(img_path)
    print(len(csv_list))
    # save
    df = pd.DataFrame(csv_list, columns=["filepath", "title"])
    print(df)
    df.to_csv(save_file_path, index=False, sep="\t")
```

## Running

Get the pre-trained model from https://huggingface.co/flashingtt/Pic2CuP

```sh
mkdir saved_models
cd saved_models
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/flashingtt/Pic2CuP
cd Pic2CuP
git pull
```

### Training

```sh train.sh```

```sh
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -u src/main.py \
	--save-frequency 1 \
	--train-data ./cc3m/Train_GCC-training_output.csv \
	--warmup 10000 \
	--batch-size 1024 \
	--lr 1e-4 \
	--wd 0.1 \
	--epochs 30 \
	--workers 8 \
	--openai-pretrained \
	--model ViT-L/14 \
	--n-ctx 3 \
	--n-img 1 \
	--pseudo-only \
	--prompt learnable \
	--meta_prompt \
```

### Validation

* ImageNet

```sh val_imgnet.sh```

```sh
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

# evaluation on imgnet
data_name=imgnet
gpu_id=0

CUDA_VISIBLE_DEVICES=3 \
python src/eval_retrieval.py \
    --openai-pretrained \
    --saved-model-path ./logs/lr=0.0001_wd={args.wd}_agg={args.aggregate}_model={args.model}_batchsize={args.batch_size}_workers={args.workers}_date=2024-xx-xx-xx-xx-xx/checkpoints \
    --eval-mode $data_name \
    --gpu $gpu_id \
    --model ViT-L/14 \
    --batch-size 1024 \
    --prompt learnable \
    --meta_prompt \
    --date '20240227' \
    --target-pad \
    --val-result-txt ./logs/lr=0.0001_wd={args.wd}_agg={args.aggregate}_model={args.model}_batchsize={args.batch_size}_workers={args.workers}_date=2024-xx-xx-xx-xx-xx/recall_result_imgnet.txt
```

* COCO

```sh val_coco.sh```

```sh
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

# evaluation on coco
data_name=coco
gpu_id=0

CUDA_VISIBLE_DEVICES=2 \
python src/eval_retrieval.py \
    --openai-pretrained \
    --date '20240227' \
    --saved-model-path ./logs/lr=0.0001_wd={args.wd}_agg={args.aggregate}_model={args.model}_batchsize={args.batch_size}_workers={args.workers}_date=2024-xx-xx-xx-xx-xx/checkpoints \
    --eval-mode $data_name \
    --gpu $gpu_id \
    --model ViT-L/14 \
    --batch-size 1024 \
    --prompt learnable \
    --meta_prompt \
    --target-pad \
    --val-result-txt ./logs/lr=0.0001_wd={args.wd}_agg={args.aggregate}_model={args.model}_batchsize={args.batch_size}_workers={args.workers}_date=2024-xx-xx-xx-xx-xx/recall_result_coco.txt
```

* CIRR

```sh val_cirr.sh```

```sh
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

# evaluation on cirr
data_name=cirr
gpu_id=0

CUDA_VISIBLE_DEVICES=3 \
python src/eval_retrieval.py \
    --openai-pretrained \
    --date '20240227' \
    --saved-model-path ./logs/lr=0.0001_wd={args.wd}_agg={args.aggregate}_model={args.model}_batchsize={args.batch_size}_workers={args.workers}_date=2024-xx-xx-xx-xx-xx/checkpoints \
    --eval-mode $data_name \
    --gpu $gpu_id \
    --model ViT-L/14 \
    --batch-size 1024 \
    --prompt learnable \
    --meta_prompt \
    --target-pad \
    --val-result-txt ./logs/lr=0.0001_wd={args.wd}_agg={args.aggregate}_model={args.model}_batchsize={args.batch_size}_workers={args.workers}_date=2024-xx-xx-xx-xx-xx/recall_result_cirr.txt
    
```

* FashionIQ

```sh val_fiq.sh```

```sh
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

# evaluation on fashioniq
cloth_type=dress
gpu_id=0

CUDA_VISIBLE_DEVICES=2 \
python src/eval_retrieval.py \
    --openai-pretrained \
    --date '20240227' \
    --saved-model-path ./logs/lr=0.0001_wd={args.wd}_agg={args.aggregate}_model={args.model}_batchsize={args.batch_size}_workers={args.workers}_date=2024-xx-xx-xx-xx-xx/checkpoints \
    --eval-mode fashion \
    --source $cloth_type \
    --gpu $gpu_id \
    --model ViT-L/14 \
    --batch-size 1024 \
    --prompt learnable \
    --meta_prompt \
    --target-pad \
    --val-result-txt ./logs/lr=0.0001_wd={args.wd}_agg={args.aggregate}_model={args.model}_batchsize={args.batch_size}_workers={args.workers}_date=2024-xx-xx-xx-xx-xx/recall_result_fiq_$cloth_type.txt
    
```