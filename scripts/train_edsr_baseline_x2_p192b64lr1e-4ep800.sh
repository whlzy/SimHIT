export CUDA_VISIBLE_DEVICES=1
exp_name='exp_edsr/edsr_baseline_x2_p192b64lr1e-4ep800'
config_path='config/exp_edsr/edsr_baseline_x2_p192b64lr1e-4ep800.yml'
python train_edsr.py --config_path $config_path --exp_name $exp_name