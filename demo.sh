export PYTHONPATH="$PYTHONPATH:$PWD/src"
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

gpu_id=2

python src/demo.py \
    --openai-pretrained \
    --resume /home/jumpserver/yxt/cir/composed_image_retrieval/logs/2024-03-04-16-07-24_best/checkpoints/epoch_30.pt \
    --retrieval-data cirr \
    --query_file /home/jumpserver/yxt/cir/datasets/cirr/dev/dev-244-0-img0.png\
    --prompts "show three bottles of soft drink"\
    --demo-out ./demo_out\
    --gpu $gpu_id \
    --model ViT-L/14 \
    --batch-size 1024 \
    --prompt learnable \
    --meta_prompt \
    # --target-pad \