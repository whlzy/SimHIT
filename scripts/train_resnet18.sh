exp_name='exp1_mlp/test_resnet'
config_path='config/exp_PlantSeedlings/test_resnet18.yml'
rm -rf exp/$exp_name
python train_resnet.py --config_path $config_path --exp_name $exp_name