export PYTHONPATH="$PYTHONPATH:$PWD/src"
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -u src/main.py \
	--save-frequency 1 \
	--train-data /mnt/vdb/yxt/huggingface/cc3m_01122022/cc/Train_GCC-training_output.csv \
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
	# --report-to wandb \
	# --transform targetpad
	# --dist-url tcp://127.0.0.1:6300 \
	# --f im2multitext
