exp_name='exp_gan/wgan_RMSprop_lr1e-3b2048ep10000'
config_path='config/exp_gan/wgan_RMSprop_lr1e-3b2048ep10000.yml'
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    train_wgan.py --config_path $config_path --exp_name $exp_name