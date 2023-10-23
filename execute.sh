echo $$
export WANDB_API_KEY=""
source environ/bin/activate
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=4 python3 main.py \
	--project c100-comparison \
	--dataset c100 \
	--valid-ratio 0.5 \
	--batch-size 256 \
	--num-workers 4 \
	--eval-batch-size 1024 \
	--autoaugment \
	--model search_mixer \
	--hidden-size  128 \
	--patch-size  4 \
	--hidden-s-candidates 16 32 64 \
	--hidden-c-candidates 128 256 512 \
	--n-cells 8 \
	--drop-p 0. \
	--clip-grad 0 \
	--cutmix-beta 1. \
	--cutmix-prob 0.5 \
	--label-smoothing 0.1 \
	--w01 1. \
	--mu 1e-4 \
	--l 1e-6 \
	--w-optimizer Adam \
	--w-lr 1e-3 \
	--w-momentum 0.9 \
	--w-weight-decay 5e-5 \
	--w-beta1 0.9 \
	--w-beta2 0.99 \
	--a-optimizer Adam \
	--a-lr  0.005 \
	--a-beta1 0.5 \
	--a-beta2 0.999 \
	--a-weight-decay 0. \
	--w-scheduler cosine \
	--epochs 300 \
	--w-min-lr 1e-6 \
	--warmup-epochs 5 \
#	--retrain-fixed \
#	--fixed-batch-size 256 \
#	--fixed-eval-batch-size 1024 \
#	--fixed-epochs 300 \
#	--fixed-lr 1e-3 \
#	--fixed-momentum 0.9 \
#	--fixed-optimizer Adam \
#	--fixed-scheduler cosine \
#	--fixed-beta1 0.9 \
#	--fixed-beta2 0.99 \
#	--fixed-weight-decay 5e-5 \
#	--fixed-label-smoothing 0.1 \
#	--fixed-warmup-epoch 5 \
#	--fixed-autoaugment \
#	--fixed-clip-grad 0 \
#	--fixed-cutmix-beta 1. \
#	--fixed-cutmix-prob 0.5 \

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=4 python3 main.py \
	--project stl10-comparison \
	--dataset stl10 \
	--valid-ratio 0.5 \
	--batch-size 256 \
	--num-workers 4 \
	--eval-batch-size 1024 \
	--autoaugment \
	--model search_mixer \
	--hidden-size  128 \
	--patch-size  4 \
	--hidden-s-candidates 16 32 64 \
	--hidden-c-candidates 128 256 512 \
	--n-cells 8 \
	--drop-p 0. \
	--clip-grad 0 \
	--cutmix-beta 1. \
	--cutmix-prob 0.5 \
	--label-smoothing 0.1 \
	--w01 1. \
	--mu 1e-4 \
	--l 1e-6 \
	--w-optimizer Adam \
	--w-lr 1e-3 \
	--w-momentum 0.9 \
	--w-weight-decay 5e-5 \
	--w-beta1 0.9 \
	--w-beta2 0.99 \
	--a-optimizer Adam \
	--a-lr  0.005 \
	--a-beta1 0.5 \
	--a-beta2 0.999 \
	--a-weight-decay 0. \
	--w-scheduler cosine \
	--epochs 300 \
	--w-min-lr 1e-6 \
	--warmup-epochs 5 \

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=4 python3 main.py \
	--project svhn-comparison \
	--dataset svhn \
	--valid-ratio 0.5 \
	--batch-size 256 \
	--num-workers 4 \
	--eval-batch-size 1024 \
	--autoaugment \
	--model search_mixer \
	--hidden-size  128 \
	--patch-size  4 \
	--hidden-s-candidates 16 32 64 \
	--hidden-c-candidates 128 256 512 \
	--n-cells 8 \
	--drop-p 0. \
	--clip-grad 0 \
	--cutmix-beta 1. \
	--cutmix-prob 0.5 \
	--label-smoothing 0.1 \
	--w01 1. \
	--mu 1e-4 \
	--l 1e-6 \
	--w-optimizer Adam \
	--w-lr 1e-3 \
	--w-momentum 0.9 \
	--w-weight-decay 5e-5 \
	--w-beta1 0.9 \
	--w-beta2 0.99 \
	--a-optimizer Adam \
	--a-lr  0.005 \
	--a-beta1 0.5 \
	--a-beta2 0.999 \
	--a-weight-decay 0. \
	--w-scheduler cosine \
	--epochs 300 \
	--w-min-lr 1e-6 \
	--warmup-epochs 5 \

