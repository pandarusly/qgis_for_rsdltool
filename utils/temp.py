import os
import sys

import torch
from omegaconf import OmegaConf


def load_jit_model(MODEL_PATH, device, MODEL_CFG=None, half_infer=False):
    print("---------start load_model!------------------")
    print(MODEL_PATH)
    if MODEL_CFG is not None:
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname((abspath))
        dname = os.path.dirname((dname))
        sys.path.append(dname)
        import hydra
        # MODEL_CFG['Config']['MODEL']['pretrained'] = None
        MODEL_CFG['Config']['MODEL']['backbone']['init_cfg'] = None
        model = hydra.utils.instantiate(MODEL_CFG, _convert_="all")
        ckpt = torch.load(MODEL_PATH, map_location=device)
        if isinstance(ckpt, dict):
            ckpt = ckpt["state_dict"]
        model.load_state_dict(ckpt)  # ["state_dict"]
        del ckpt
    else:
        # model = torch.jit.load(MODEL_PATH)
        model = torch.load(MODEL_PATH)
    if half_infer:
        model.half()
    model.to(device)
    model.eval()

    print("---------complete load_model!------------------")

    return model, device


MODEL_PATH = r'F:\study-note\python-note\Advance_py\DLDEV\Task-36869338\代码截图\代码截图\论文代码\log\LeVirCd\MDACN\2022-07-25_11-45-23\checkpoints\mdac_f10.9227.ckpt'
CFG_PATH = r'F:\study-note\python-note\Advance_py\DLDEV\Task-36869338\代码截图\代码截图\论文代码\log\LeVirCd\MDACN\2022-07-25_11-45-23\tensorboard\da\hparams.yaml'
MODEL_CFG = OmegaConf.to_object(OmegaConf.load(CFG_PATH).model)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model, device = load_jit_model(MODEL_PATH, device, MODEL_CFG, half_infer=False)
from trainers.modules.BaseChange import BaseChangeLite
