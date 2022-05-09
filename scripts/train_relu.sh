exp_name='exp1_mlp/test_relu'
config_path='config/exp1_mlp/test_relu.yml'
rm -rf exp/$exp_name
python train_resnet.py --config_path $config_path --exp_name $exp_name