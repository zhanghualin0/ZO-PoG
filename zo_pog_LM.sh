export PYTHONWARNINGS="ignore"
export TRANSFORMERS_OFFLINE=0

api_limit=4000
gpu_no=$1
prompt_learning_rate=$2 #0.01
p0_learning_rate=$3 #0.001
prompt_length=$4
kshot=16
for tn in cola qnli wnli snli mnli;do
    for seed in 42 14 81;do
        command="CUDA_VISIBLE_DEVICES=$gpu_no python ./zo_pog_LM.py --task_name=$tn --per_device_train_batch_size 32 --per_device_eval_batch_size 128 --seed=$seed --k_shot $kshot --prompt_learning_rate $prompt_learning_rate --p0_learning_rate $p0_learning_rate --n_prompt_tokens $prompt_length --api_limit $api_limit --loss_type ce"
        bash -c "$command"
    done
done