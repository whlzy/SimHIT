exp_name='test_hardswish'
config_path='config/test_hardswish.yml'
rm -rf exp/$exp_name
python train.py --config_path $config_path --exp_name $exp_name