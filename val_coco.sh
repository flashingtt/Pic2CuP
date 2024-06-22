export PYTHONPATH="$PYTHONPATH:$PWD/src"
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

# evaluation on coco
data_name=coco
gpu_id=0

CUDA_VISIBLE_DEVICES=2 \
python src/eval_retrieval.py \
    --openai-pretrained \
    --date '20240227' \
    --saved-model-path /home/jumpserver/yxt/cir/composed_image_retrieval/logs/lr=0.0001_wd={args.wd}_agg={args.aggregate}_model={args.model}_batchsize={args.batch_size}_workers={args.workers}_date=2024-03-14-01-22-29/checkpoints \
    --eval-mode $data_name \
    --gpu $gpu_id \
    --model ViT-L/14 \
    --batch-size 1024 \
    --prompt learnable \
    --meta_prompt \
    --target-pad \
    --val-result-txt /home/jumpserver/yxt/cir/composed_image_retrieval/logs/lr=0.0001_wd={args.wd}_agg={args.aggregate}_model={args.model}_batchsize={args.batch_size}_workers={args.workers}_date=2024-03-14-01-22-29/recall_result_coco.txt
    

# visualization
# CUDA_VISIBLE_DEVICES=1 \
# python src/eval_retrieval.py \
#     --openai-pretrained \
#     --date '20240227' \
#     --resume /home/jumpserver/yxt/cir/composed_image_retrieval/logs/2024-03-04-16-07-24_best/checkpoints/epoch_30.pt \
#     --eval-mode $data_name \
#     --gpu $gpu_id \
#     --model ViT-L/14 \
#     --batch-size 1024 \
#     --prompt learnable \
#     --meta_prompt \
#     --target-pad \

    # --visual-compared \
    # --best-visual-path /home/jumpserver/yxt/cir/composed_image_retrieval/logs/2024-03-04-16-07-24_best/visual_result_best/coco.txt
    