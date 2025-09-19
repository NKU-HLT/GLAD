#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

export MKL_INTERFACE_LAYER=LP64,GNU
export CONDA_MKL_INTERFACE_LAYER_BACKUP=${MKL_INTERFACE_LAYER}
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export QT_XCB_GL_INTEGRATION=${QT_XCB_GL_INTEGRATION:-software}

stage=11
stop_stage=13

# Settings
tag=$(echo $(basename $0) | cut -d "_" -f 2 | cut -d "." -f 1)  # read file name

asr_train_tag=$tag
asr_stats_tag=subA
lm_tag=subA

asr_config=configs/glad_exp/glad.yaml
train_set=train_960_sub_1mix2mix
valid_set="dev_clean_2mix"
test_sets="test_clean_1mix test_clean_2mix test_clean_2mix"
inference_config=configs/decode/decode_asr_aed_only.yaml
inference_asr_model=valid.acc.ave_10best.pth

use_lm=false
use_wordlm=false
gpu_inference=false

nj=64
ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

# Run
./asr.sh                                               \
    --nj ${nj}                                         \
    --inference_nj ${nj}                               \
    --gpu_inference ${gpu_inference}                               \
    --ngpu ${ngpu}                                     \
    --stage ${stage}                                   \
    --stop_stage ${stop_stage}                         \
    --asr_exp exp/asr_train_asr_${asr_train_tag} \
    --asr_stats_dir exp/asr_stats_${asr_stats_tag}    \
    --lm_exp exp/lm_train_lm_${lm_tag} \
    --lm_stats_dir exp/lm_stats_${lm_tag} \
    --lang en                                          \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type bpe                                  \
    --bpe_nlsyms "$" \
    --nbpe 5000 \
    --bpe_train_text "data/train_960_1mix/text" \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --inference_asr_model "${inference_asr_model}"     \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_speech_fold_length 1024 \
    --asr_text_fold_length 600 \
    --lm_fold_length 600 \
    --lm_train_text "data/${train_set}/text" "$@"
