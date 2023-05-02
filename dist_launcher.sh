CONFIG=$1
GPUS=$2
PORT=${PORT:-28511}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch \
                                --nproc_per_node=$GPUS \
                                --master_port=$PORT \
                                $(dirname "$0")/dist_train_dev.py $CONFIG \
                                --launcher pytorch ${@:3} \
                                --deterministic