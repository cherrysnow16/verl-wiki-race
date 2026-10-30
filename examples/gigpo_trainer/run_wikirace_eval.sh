set -x

# 학습 없이 easy split 전체(200게임) 평가만 수행 (val_only).
# 모델은 MODEL 환경변수로 지정한다:
#   MODEL=meta-llama/Llama-3.2-1B-Instruct bash examples/gigpo_trainer/run_wikirace_eval.sh
ENGINE=${ENGINE:-vllm}
MODEL=${MODEL:-meta-llama/Llama-3.2-3B-Instruct}

num_cpus_per_env_worker=0.1

# easy: 200게임 전체를 평가 (val_train_ratio=0.0 → val pool = 전체).
train_data_size=8
val_data_size=200

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=10000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=12288 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.gamma=0.95 \
    env.env_name=wikirace/easy \
    env.seed=0 \
    env.max_steps=30 \
    env.rollout.n=1 \
    env.history_length=2 \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.wikirace.difficulty=easy \
    env.wikirace.max_choices=50 \
    env.wikirace.train_ratio=0.8 \
    +env.wikirace.val_train_ratio=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_wikirace' \
    trainer.experiment_name="eval_easy_$(basename $MODEL)" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    trainer.val_only=True $@
