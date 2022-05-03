# PR_Experiment of MNIST
A simple experiment of MNIST for HIT PR Experiment.

## Results
|Dataset |Model  |Accuracy |
|:-:| :----: | :----:|
|MNIST |Hardswish | 98.16%|
|MNIST |ReLU | 88.35%|

## Train and Eval
- Clone
```
git clone https://github.com/whlzy/PR_EXP_1_MNIST.git
cd PR_EXP_1_MNIST
```

- You should modify the path of **test_hardswish.yml** or **config/test_relu.yml** to your local dataset path.
```
cd config
cat test_relu.yml
...
...
```

- You can choose gpu or cpu in the **test_hardswish.yml** or **config/test_relu.yml**.

- You can run the script.
```
cd ..
sh scripts/train_hardswish.sh
sh scripts/train_relu.sh
```

- You can change the net config in the **config/test_hardswish.yml** or **config/test_relu.yml**.

- You can add some new experiments with just adding new script in **scripts** and new yaml file in **config**.

- You can add new dataset in **data**, but maybe need to change some codes.

## EXP Log
EXP log is in the exp/test_hardswish and exp/test_relu.

The output.log is logged by **mmcv logging**.

The tensorboard log is in the **exp/*/logdir**.

The best checkpoint is in the **exp/*/checkpoint/best/model_best.pth**.