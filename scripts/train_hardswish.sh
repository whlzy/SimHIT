exp_name='exp1_mlp/test_hardswish'
config_path='config/exp1_mlp/test_hardswish.yml'
rm -rf exp/$exp_name
python train_mlp.py --config_path $config_path --exp_name $exp_name