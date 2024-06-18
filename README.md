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

1. Get CC3M from 

## Running

### Training

```sh
sh train.sh
```

### Validation

