CONFIG=$1
GPUS=$2
PORT=${PORT:-28514}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES="3" python -m torch.distributed.launch \
                                --nproc_per_node=$GPUS \
                                --master_port=$PORT \
                                $(dirname "$0")/dist_train.py $CONFIG \
                                --launcher pytorch ${@:3} \
                                --deterministic