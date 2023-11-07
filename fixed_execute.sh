echo $$
source environ/bin/activate
export TORCH_DISTRIBUTED_DEBUG=DETAIL
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3,2 python3 fixed_main.py \
	--path-to-supernet /home/tgadea/MLP-NAS/out/experiment_2023-10-26_18:44:01 \
	--model fixed-mixer \
	--dataset imagenet \
	--img-size 224 \
	--batch-size 128 \
	--eval-batch-size 128 \
	--valid-ratio 0. \
	--epochs 1 \
	--lr 1e-3 \
	--momentum 0.9 \
	--optimizer Adam \
	--scheduler cosine \
	--beta1 0.9 \
	--beta2 0.99 \
	--weight-decay 5e-5 \
	--label-smoothing 0.1 \
	--warmup-epoch 5 \
	--clip-grad 0 \
	--cutmix-beta 1. \
	--cutmix-prob 0.5 \
	--distributed \
	--recovery-interval 0 \
	--use-timm-transform \
	--hflip 0.5 \
	--vflip 0. \
	--color-jitter 0.4 \
	--train-interpolation bilinear \
	--reprob 0. \
	--remode pixel \
	--recount 1 \
	--autoaugment v0 \
	--th-arch 0.5 \
	--binarize-arch \
	--verbose
