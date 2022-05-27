import os
import sys
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch
import argparse
import model.resnet as resnet

convertargs = argparse.ArgumentParser()
convertargs.add_argument('--num_classes', type=int, help='the number of classes')
convertargs = convertargs.parse_args()

path = "pretrain/mocov2_resnet50_8xb32-coslr-200e_in1k_20220225-89e03af4.pth"
def resnet50(num_classes=None):
    return resnet.ResNet(block="Bottleneck", layers=[3, 4, 6, 3], num_classes=num_classes)

if __name__ == '__main__':
    ResNet50 = resnet50(convertargs.num_classes)
    state_dict = torch.load(path)
    new_state_dict = state_dict["state_dict"]
    our_state_dict = ResNet50.state_dict()
    for k in new_state_dict.keys():
        if k in our_state_dict.keys() and not k.startswith('fc'):
            our_state_dict[k] = new_state_dict[k]
    ResNet50.load_state_dict(our_state_dict)
    save_model = {"author":"whlzy", "model":our_state_dict}
    torch.save(save_model, "pretrain/mocov2_resnet50_8xb32-coslr-200e_in1k_simhit.pth")








