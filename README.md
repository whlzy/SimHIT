# 🎈 SimHIT: A Simple Framework for HIT Pattern Recognition Experiment 🎈
## Train and Eval 🚀
- Clone
```
git clone git@github.com:whlzy/SimHIT.git
cd PR_EXP
```

- You should modify the path of **config/exp_mlp/test_hardswish.yml** or **config/exp_mlp/test_relu.yml** to your local dataset path.
```
cd config/exp_mlp
cat test_relu.yml
...
...
```

- You can choose gpu or cpu in the **config/exp_mlp/test_hardswish.yml** or **config/exp_mlp/test_relu.yml**.

- You can run the script.
```
cd ../..
sh scripts/train_hardswish.sh
sh scripts/train_relu.sh
```

- You can change the net config in the **config/exp_mlp/test_hardswish.yml** or **config/exp_mlp/test_relu.yml**.

- You can add some new experiments with just adding new script in **scripts** and new yaml file in **config**.

- You can add new dataset in **src/data**, but maybe need to change some codes.

- You can rewrite a new training code like **train_mlp.py** using **src/runner.py**. **src/runner.py** is a class which assembles partial training process and config process. You just need to use the **src/runner.py** and rewrite *set_data*, *set_model*, *train_one_epoch* and *test_one_epoch* like **train_mlp.py**. Like the **train_mlp.py**, you can freely modify the network and modify the training process in *train_one_epoch*.

## EXP Log 📖
EXP log is in the **exp/*/test_hardswish**.

The output.log is logged by **mmcv logging** in the **exp/*/test_hardswish/output.log**.

The config you used is written in the **exp/*/test_hardswish/config.yaml**.

The tensorboard log is in the **exp/*/test_hardswish/logdir**.

The best checkpoint is in the **exp/*/test_hardswish/checkpoint/best/model_best.pth**.

## PR_Experiment ⚡
###
- train mlp.
```
sh scripts/train_hardswish.sh
sh scripts/train_relu.sh
```
- train alexnet.
```
sh scripts/train_AlexNet.sh
```
- train resnet18.
```
sh scripts/train_resnet18.sh
```
### Results 📊
|Dataset |Model  |Accuracy | Tip |
|:-:|:----:|:----:|:----:|
|MNIST |Hardswish | 98.16%|  |
|MNIST |ReLU | 88.35%|  |
|Caltech101 |AlexNet | 71.46%| train data : val data = 9 : 1 |
|PlantSeedlings |ResNet18 | 94.1909%| train data : val data = 9 : 1 |

## License ⭐
This project is released under the [Apache 2.0 license](https://github.com/whlzy/PR_EXP/blob/master/LICENSE).
