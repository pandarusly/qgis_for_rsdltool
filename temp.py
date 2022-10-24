
import torch
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf


def get_benchmarks_flops(model, input_shape, input_constructor=lambda x: {"img1": torch.ones(()).new_empty((1, *x)),
                                                                          "img2": torch.ones(()).new_empty((1, *x))}):
    from mmcv.cnn import get_model_complexity_info
    flops, params = get_model_complexity_info(model, input_shape, input_constructor=input_constructor,
                                              print_per_layer_stat=False)
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


def init_model(cfg):

    model = instantiate(cfg.model, _convert_="all")
    input_shape = (3, 256, 256)
    get_benchmarks_flops(model, input_shape=input_shape)
    return None


@hydra.main(config_path="configs/", config_name="train.yaml")
def test_model(cfg):
    # python temp.py model=benchmarks/pdacn
    init_model(cfg)


if __name__ == "__main__":
    from trainers.modules.BaseChange import BaseChangeLite
    from trainers.modules.MMCHANGE import MMCHANGE
