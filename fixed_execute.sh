echo $$
source wandb.key.sh
source environ/bin/activate
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=4 python3 fixed_main.py \
	--project mixer-retrain-fixed \
	--model mlp-mixer \
	--path /home/tgadea/mixer-NAS/out/mixer-retrain-fixed/experiment_2023-09-16_01:44:29 \
	--fixed-batch-size 256 \
	--fixed-eval-batch-size 1024 \
	--fixed-epochs 300 \
	--fixed-lr 1e-3 \
	--fixed-momentum 0.9 \
	--fixed-optimizer Adam \
	--fixed-scheduler cosine \
	--fixed-beta1 0.9 \
	--fixed-beta2 0.99 \
	--fixed-weight-decay 5e-5 \
	--fixed-label-smoothing 0.1 \
	--fixed-warmup-epoch 5 \
	--fixed-autoaugment \
	--fixed-clip-grad 0 \
	--fixed-cutmix-beta 1. \
	--fixed-cutmix-prob 0.5 \
