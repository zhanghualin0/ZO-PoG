export PYTHONWARNINGS="ignore"
export TRANSFORMERS_OFFLINE=0

api_limit=2000 #1000
gpu_no=$1
prompt_learning_rate=$2 #0.01
p0_learning_rate=$3 #0.001
prompt_length=$4
kshot=16
model=gpt2-xl #llama3
batch_size=16 #8
for tn in mnli qnli snli wnli cola;do
    for seed in 42 14 81;do
        command="CUDA_VISIBLE_DEVICES=$gpu_no python ./zo_pog_causal.py --model_name_or_path $model --task_name=$tn --per_device_train_batch_size $batch_size --per_device_eval_batch_size $batch_size --seed=$seed --k_shot $kshot --prompt_learning_rate $prompt_learning_rate --p0_learning_rate $p0_learning_rate --n_prompt_tokens $prompt_length --api_limit $api_limit --loss_type ce"
        bash -c "$command"
    done
done
