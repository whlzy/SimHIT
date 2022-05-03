exp_name='test_relu'
config_path='config/test_relu.yml'
rm -rf exp/$exp_name
python train.py --config_path $config_path --exp_name $exp_name