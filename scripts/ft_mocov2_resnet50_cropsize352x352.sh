exp_name='exp_plantseedlings/ft_mocov2_resnet50'
config_path='config/exp_plantseedlings/test_resnet50_ft_mocov2_cropsize352x352.yml'
python src/tools/convert_mmself_mocov2_resnet50_to_SimHIT.py --num_classes 12 &&
python train_mocov2_resnet50_ft.py --config_path $config_path --exp_name $exp_name