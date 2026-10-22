#!/bin/sh

cd $(dirname $(dirname "$0")) || exit
ROOT_DIR=$(pwd)
PYTHON=python

TRAIN_CODE=train.py

DATASET=nuscenes
CONFIG="None"
EXP_NAME=debug
WEIGHT="None"
RESUME=false
NUM_GPU=None
NUM_MACHINE=1
DIST_URL="auto"
CUDA_DEVICES="0"
KD_LOSS_TYPE_1="" # 'logits'
KD_LOSS_TYPE_2="" # 'feat'
KD_LOSS_TYPE_1_WEIGHT=1
KD_LOSS_TYPE_2_WEIGHT=1
KD_TEMPERATURE=1
CWKD=false
DIFFKD=false
DIFFKD_MODE=""
M_P2P=false
NOISE_W=1
ALIGN_W=1
FEAT_VIS=false
FEAT_VIS_MAX_TOTAL=100
FEAT_VIS_MAX_PER_EPOCH=4

while getopts "p:d:c:n:w:g:m:r:v:a:b:e:f:t:W:A:M:DklVF:P:" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    d)
      DATASET=$OPTARG
      ;;
    c)
      CONFIG=$OPTARG
      ;;
    n)
      EXP_NAME=$OPTARG
      ;;
    w)
      WEIGHT=$OPTARG
      ;;
    r)
      RESUME=$OPTARG
      ;;
    g)
      NUM_GPU=$OPTARG
      ;;
    m)
      NUM_MACHINE=$OPTARG
      ;;
    v)
      CUDA_DEVICES=$OPTARG
      ;;
    a)
      KD_LOSS_TYPE_1=$OPTARG
      ;;
    b)
      KD_LOSS_TYPE_2=$OPTARG
      ;;
    e)
      KD_LOSS_TYPE_1_WEIGHT=$OPTARG
      ;;
    f)
      KD_LOSS_TYPE_2_WEIGHT=$OPTARG
      ;;
    t)
      KD_TEMPERATURE=$OPTARG
      ;;
    W)
      NOISE_W=$OPTARG
      ;;
    A)
      ALIGN_W=$OPTARG
      ;;
    k)
      CWKD=true
      ;;
    D)
      DIFFKD=true
      ;;
    M)
      DIFFKD_MODE=$OPTARG
      ;;
    l)
      M_P2P=true
      ;;
    V)
      FEAT_VIS=true
      ;;
    F)
      FEAT_VIS_MAX_TOTAL=$OPTARG
      ;;
    P)
      FEAT_VIS_MAX_PER_EPOCH=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

if [ "${NUM_GPU}" = 'None' ]
then
  NUM_GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
fi

if [ -n "$DIFFKD_MODE" ]; then
  DIFFKD_MODE_FLAG="--diffkd-training-mode $DIFFKD_MODE"
else
  DIFFKD_MODE_FLAG=""
fi

echo "Experiment name: $EXP_NAME"
echo "Python interpreter dir: $PYTHON"
echo "Dataset: $DATASET"
echo "Config: $CONFIG"
echo "GPU Num: $NUM_GPU"
echo "Machine Num: $NUM_MACHINE"

if [ -n "$SLURM_NODELIST" ]; then
  MASTER_HOSTNAME=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
  MASTER_ADDR=$(getent hosts "$MASTER_HOSTNAME" | awk '{ print $1 }')
  MASTER_PORT=$((10000 + 0x$(echo -n "${DATASET}/${EXP_NAME}" | md5sum | cut -c 1-4 | awk '{print $1}') % 20000))
  DIST_URL=tcp://$MASTER_ADDR:$MASTER_PORT
fi

echo "Dist URL: $DIST_URL"

if [ "$CWKD" = true ]; then
  CWKD_FLAG="--cwkd"   # 传给 python 的
  CWKD_TAG="_cwkd"     # 用在目录名里的
else
  CWKD_FLAG=""
  CWKD_TAG=""
fi

if [ "$DIFFKD" = true ]; then
  DIFFKD_FLAG="--diffkd"
  DIFFKD_TAG="_diffkd"
else
  DIFFKD_FLAG=""
  DIFFKD_TAG=""
fi

if [ "$M_P2P" = true ]; then
  M_P2P_FLAG="--m-p2p-enable"
  M_P2P_TAG="_mp2p"
else
  M_P2P_FLAG=""
  M_P2P_TAG=""
fi

if [ "$FEAT_VIS" = true ]; then
  FEAT_VIS_FLAG="--feat-vis --feat-vis-max-total $FEAT_VIS_MAX_TOTAL --feat-vis-max-per-epoch $FEAT_VIS_MAX_PER_EPOCH"
else
  FEAT_VIS_FLAG=""
fi

EXP_DIR=exp/${DATASET}/${EXP_NAME}${DIFFKD_TAG}${CWKD_TAG}${M_P2P_TAG}/${KD_LOSS_TYPE_1}_wei1:${KD_LOSS_TYPE_1_WEIGHT}tem1_${KD_TEMPERATURE}_${KD_LOSS_TYPE_2}_${KD_LOSS_TYPE_2_WEIGHT}_nw${NOISE_W}_aw${ALIGN_W}
MODEL_DIR=${EXP_DIR}/model
CODE_DIR=${EXP_DIR}/code
CONFIG_DIR=configs/${DATASET}/${CONFIG}.py


echo " =========> CREATE EXP DIR <========="
echo "Experiment dir: $ROOT_DIR/$EXP_DIR"
if [ "${RESUME}" = true ] && [ -d "$EXP_DIR" ]
then
  CONFIG_DIR=${EXP_DIR}/config.py
  WEIGHT=$MODEL_DIR/model_last.pth
else
  RESUME=false
  mkdir -p "$MODEL_DIR" "$CODE_DIR"
  cp -r scripts tools pointcept "$CODE_DIR"
fi

echo "Loading config in:" $CONFIG_DIR
export PYTHONPATH=./$CODE_DIR
echo "Running code in: $CODE_DIR"


echo " =========> RUN TASK <========="
# nvidia-smi 按 PCI 编号，CUDA 默认按算力排序。3 号是 RTX PRO 6000，
# 当前 PyTorch 不支持它，默认顺序会把它放到最后，4 号及之后的卡会整体错一位。
# 固定成 PCI 顺序后，-v 的编号与 nvidia-smi 一致。
export CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"
echo "CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER"
ulimit -n 65536
if [ "${WEIGHT}" = "None" ]
then
    # echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES $PYTHON $CODE_DIR/tools/$TRAIN_CODE --config-file $CONFIG_DIR --num-gpus $NUM_GPU --num-machines $NUM_MACHINE --machine-rank ${SLURM_NODEID:-0} --dist-url ${DIST_URL} --kd-loss-type-1 $KD_LOSS_TYPE_1 --kd-loss-type-2 $KD_LOSS_TYPE_2 --kd-loss-weight-1 $KD_LOSS_TYPE_1_WEIGHT --kd-loss-weight-2 $KD_LOSS_TYPE_2_WEIGHT --options save_path=$EXP_DIR"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$NUM_GPU" \
    --num-machines "$NUM_MACHINE" \
    --machine-rank ${SLURM_NODEID:-0} \
    --dist-url ${DIST_URL} \
    --kd-loss-type-1 "$KD_LOSS_TYPE_1" \
    --kd-loss-type-2 "$KD_LOSS_TYPE_2" \
    --kd-loss-weight-1 "$KD_LOSS_TYPE_1_WEIGHT" \
    --kd-loss-weight-2 "$KD_LOSS_TYPE_2_WEIGHT" \
    --kd_temperature "$KD_TEMPERATURE" \
    --noise-w "$NOISE_W" \
    --align-w "$ALIGN_W" \
    $DIFFKD_FLAG \
    $DIFFKD_MODE_FLAG \
    $CWKD_FLAG \
    $M_P2P_FLAG \
    $FEAT_VIS_FLAG \
    --options save_path="$EXP_DIR"
else
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$NUM_GPU" \
    --num-machines "$NUM_MACHINE" \
    --machine-rank ${SLURM_NODEID:-0} \
    --dist-url ${DIST_URL} \
    --kd-loss-type-1 "$KD_LOSS_TYPE_1" \
    --kd-loss-type-2 "$KD_LOSS_TYPE_2" \
    --kd-loss-weight-1 "$KD_LOSS_TYPE_1_WEIGHT" \
    --kd-loss-weight-2 "$KD_LOSS_TYPE_2_WEIGHT" \
    --kd_temperature "$KD_TEMPERATURE" \
    --noise-w "$NOISE_W" \
    --align-w "$ALIGN_W" \
    $DIFFKD_FLAG \
    $DIFFKD_MODE_FLAG \
    $CWKD_FLAG \
    $M_P2P_FLAG \
    $FEAT_VIS_FLAG \
    --options save_path="$EXP_DIR" resume="$RESUME" weight="$WEIGHT"
fi
