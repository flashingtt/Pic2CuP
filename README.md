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

1. Get CC3M from https://huggingface.co/datasets/yxchng/cc3m_01122022

2. Go to https://ai.google.com/research/ConceptualCaptions/download and press download button to get a 500MB .tsv file

3. Download datasets

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/yxchng/cc3m_01122022
cd cc3m_01122022
git lfs pull --include="*.tar"
tar -xvf *.tar
```

## Running

### Training

```sh
sh train.sh
```

### Validation

