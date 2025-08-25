python train.py \
    --window 120 \
    --device 1 \
    --log_interval 100 \
    --input_dim 9 \
    --embed_dim 128 \
    --depth 8 \
    --batch_size 96 \
    --train_num_steps 200000 \
    --gradient_accumulate_every 1 \
    --exp_dir="Experiments/exp10" \
    --learning_rate 0.0002 \
    --save_and_sample_every 10000
    