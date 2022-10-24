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
        sys.path.append(abspath)
        sys.path.append(dname)
        print(dname)
        import hydra
        if MODEL_CFG['Config']['MODEL'].get('backbone'):
            MODEL_CFG['Config']['MODEL']['backbone']['init_cfg'] = None
        else:
            MODEL_CFG['Config']['MODEL']['pretrained'] = None
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


MODEL_PATH = r'C:\Users\admin\Pictures\权重\雄安建筑物数据集\UVA\UVAtrain1\checkpoints\epoch_066_f10.777.ckpt'
CFG_PATH = r'C:\Users\admin\Pictures\权重\雄安建筑物数据集\UVA\UVAtrain1\csv\uva_0809_building\hparams.yaml'
MODEL_CFG = OmegaConf.to_object(OmegaConf.load(CFG_PATH).model)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model, device = load_jit_model(MODEL_PATH, device, MODEL_CFG, half_infer=False)

x = torch.randn(1, 3, 256, 256).to(device)

pred = model(x, x)
print(pred.shape)

# C:\Users\admin\Pictures\权重\雄安建筑物数据集\UVA\UVAtrain1\checkpoints\epoch_066_f10.777.ckpt
# C:\Users\admin\Pictures\权重\雄安建筑物数据集\UVA\UVAtrain1\csv\uva_0809_building\hparams.yaml
