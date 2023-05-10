# CONFIG=$1
# GPUS=$2
# PORT=${PORT:-28519}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# CUDA_VISIBLE_DEVICES="2" python -m torch.distributed.launch \
#                                 --nproc_per_node=$GPUS \
#                                 --master_port=$PORT \
#                                 $(dirname "$0")/evaluation/occ_eval.py $CONFIG \
#                                 --launcher pytorch ${@:3}

CONFIG=$1
GPUS=$2
PORT=${PORT:-28515}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES="1" python -m torch.distributed.launch \
                                --nproc_per_node=$GPUS \
                                --master_port=$PORT \
                                $(dirname "$0")/test.py $CONFIG \
                                --launcher pytorch ${@:3} \
                                --deterministic