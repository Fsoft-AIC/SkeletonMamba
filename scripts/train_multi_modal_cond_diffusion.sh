python trainer_ego_music_cond_motion_diffusion.py \
    --window=120 \
    --device 2 \
    --log_interval 100 \
    --input_dim 9 \
    --embed_dim 128 \
    --depth 8 \
    --batch_size 64 \
    --train_num_steps 500000 \
    --gradient_accumulate_every 1 \
    --exp_dir="Experiments/exp1" \
    --learning_rate 0.0002 \
    --save_and_sample_every 5000
    