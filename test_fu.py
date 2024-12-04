import logging
import os.path
import hydra
from hydra.utils import get_original_cwd
import numpy as np
import torch
from utils.models import FullyConnectedNetwork
import utils.fu as fu
import utils.xiaobo as xiaobo
import matplotlib.pyplot as plt

@hydra.main(version_base=None, config_path="./conf", config_name="evaluation_conf")
def evaluation_setup(cfg):
    model_conf = cfg["model_conf"]
    equation_conf = cfg["equation_conf"]
    device = torch.device("cuda")

    conf=equation_conf['Burgers']
    weight_path = "F:/StudyNote/PINN_e/DMIS-main/pretrain/Burgers/PINN-DMIS/best.pth"
    model_conf["layer"]["layer_n"] = conf["layer_n"]
    model_conf["layer"]["layer_size"] = conf["layer_size"]
    model_conf["dim"]["output_dim"] = conf["output_dim"]
    model = FullyConnectedNetwork(model_conf).to(device)
    model.load_state_dict(torch.load(weight_path))
    a,b=xiaobo.fusample(model,16000)
    print(b.shape)
    plt.scatter(b[:,0],b[:,1])
    plt.show()
evaluation_setup()