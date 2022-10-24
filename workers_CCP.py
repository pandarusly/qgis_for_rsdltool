import os
import tempfile
from typing import Dict, Any, Union, List


from PyQt5.QtCore import QObject, pyqtSignal
from omegaconf import OmegaConf
import processing
from processing.core.Processing import Processing
from .config import InferConfig


# from .utils import del_dir, get_path_from_yaml, rasterize, rasterizeMutiCls, splitting, full_flow


def clip_raster_by_mask(temp_dir, a_ras_path, mkdir_p, mask_path=r''):
    # ------------------------- clip a with mask return tif_name
    temp_clip_dir_a = os.path.join(temp_dir, "processing_tfYpER", os.urandom(32).hex())
    mkdir_p(temp_clip_dir_a)
    temp_clip_name_a = os.path.join(temp_clip_dir_a, "OUTPUT.tif")
    params_a = {'INPUT': a_ras_path,
                'MASK': mask_path, 'NODATA': None, 'ALPHA_BAND': False,
                'CROP_TO_CUTLINE': True, 'KEEP_RESOLUTION': True, 'OPTIONS': 'COMPRESS=LZW', 'DATA_TYPE': 0,
                'OUTPUT': temp_clip_name_a}
    Processing.initialize()
    result = processing.run('gdal:cliprasterbymasklayer', params_a)
    return temp_clip_name_a


def diag_spliting(a_ras_path, b_ras_path, vec_path, mask_path, datasetOutDir, classCfgPath,
                  SplittingBlockSize, SplittingStrideSize, uniform_name, mkdir_p,
                  temp_dir, color_text_path, youhua
                  ):
    from .utils.rasterize import rasterize
    from .utils.splitting import splitting
    from .utils.rasterizeMutiCls import rasterizeMutiCls
    currentrasterlayA = os.path.splitext(os.path.basename(a_ras_path))[0]
    if mask_path is not None:  # if b_ras_path vec_path classCfgPath mask_path
        if not os.path.exists(mask_path):
            print("mask layer doesn't exist")
            return
        else:
            a_ras_path = clip_raster_by_mask(temp_dir, a_ras_path, mkdir_p, mask_path)
            b_ras_path = clip_raster_by_mask(temp_dir, b_ras_path, mkdir_p,
                                             mask_path) if b_ras_path is not None else None

    dataset_paddle = datasetOutDir
    # dataset_paddle = os.path.join(datasetOutDir, "PaddlePaddle")
    mkdir_p(dataset_paddle)
    Ras_Paddle_path = os.path.join(dataset_paddle, "rasterized/")
    imageA_Paddle_path = os.path.join(dataset_paddle, "A/")
    imageB_Paddle_path = os.path.join(dataset_paddle, "B/")
    label_Paddle_path = os.path.join(dataset_paddle, "label/")
    muti_Lable_Paddle_path = os.path.join(dataset_paddle, "muti_label/")
    mkdir_p(Ras_Paddle_path)
    mkdir_p(imageA_Paddle_path)
    mkdir_p(imageB_Paddle_path)
    mkdir_p(label_Paddle_path)
    mkdir_p(muti_Lable_Paddle_path)

    rasterize_path = {"binary": "",
                      "muticlass": "",
                      }
    # ----- raseterize binary
    if vec_path is not None:
        if 'shp' in vec_path:
            output = os.path.join(
                Ras_Paddle_path, uniform_name + "_rasterized" + ".tif" if len(
                    uniform_name) > 0 else currentrasterlayA + "_rasterized" + ".tif"
            )  # Output Rasterized File

            rasterize(a_ras_path, vec_path, output)
        else:
            output = vec_path
            output = clip_raster_by_mask(temp_dir, output, mkdir_p, mask_path) if mask_path is not None else output
            if classCfgPath is not None:
                print(" Rasterize Only Support SHPfile!")
                return
        rasterize_path["binary"] = output

    # ----- raseterize  muticlass
    if classCfgPath is not None:
        if 'shp' not in vec_path:
            print("Rasterize Only Support SHPfile!")
            return
        # color_text_path = os.path.join(self.plugin_dir + "/utils/color.txt")
        class_cfg_path = classCfgPath
        outputRasIN = os.path.join(
            Ras_Paddle_path, uniform_name + "_1_255_rasterized" + ".tif" if len(
                uniform_name) > 0 else currentrasterlayA + "_1_255_rasterized" + ".tif")
        InsSegGDALout = os.path.join(
            Ras_Paddle_path, uniform_name + "_muti_class_rasterized" + ".tif" if len(
                uniform_name) > 0 else currentrasterlayA + "_muti_class_rasterized" + ".tif")

        rasterizeMutiCls(a_ras_path, vec_path, outputRasIN,
                         InsSegGDALout, color_text_path, class_cfg_path)
        rasterize_path["binary"] = InsSegGDALout

    splitting(
        a_ras_path,
        imageA_Paddle_path,
        "jpg",
        "JPEG",
        "",
        SplittingBlockSize,
        SplittingStrideSize,
        uniform_name if len(uniform_name) > 0 else currentrasterlayA,
        youhua=youhua
    )
    # --split post-time image
    if b_ras_path is not None:
        splitting(
            b_ras_path,
            imageB_Paddle_path,
            "jpg",
            "JPEG",
            "",
            SplittingBlockSize,
            SplittingStrideSize,
            uniform_name if len(uniform_name) > 0 else currentrasterlayA,
            youhua=youhua
        )
    # --split binary label
    if vec_path is not None:
        splitting(
            output,
            label_Paddle_path,
            "png",
            "PNG",
            "",
            SplittingBlockSize,
            SplittingStrideSize,
            uniform_name if len(uniform_name) > 0 else currentrasterlayA,
            youhua=youhua
        )  # should be the same name of image. vector name if needed-> currentvectorlay

    # --- spliting Multiclass Label
    print("split muti----------------------------------")
    if classCfgPath is not None:
        splitting(
            outputRasIN,
            muti_Lable_Paddle_path,
            "png",
            "PNG",
            "",
            SplittingBlockSize,
            SplittingStrideSize,
            uniform_name if len(uniform_name) > 0 else currentrasterlayA,
            youhua=None
        )  # should be the same name of image. vector name if needed-> currentvectorlay

    return rasterize_path


def create_splitting(
        a_series_path,
        b_series_path,
        v_series_path,
        a_ras_path,
        b_ras_path,
        vec_path,
        mask_path,
        classCfgPath,
        datasetOutDir,
        SplittingBlockSize,
        SplittingStrideSize,
        uniform_name,
        useBatchCheck,
        checkBoxSB_2,
        useChangeCheck,
        temp_dir, color_text_path,
        youhua
):
    def mkdir_p(path):
        if not os.path.exists(path):
            os.makedirs(path)

    show_list = []
    if not useBatchCheck:
        print('processing {}/{} file {} ...'.format(1, 1, a_ras_path))
        rasterize_path = diag_spliting(a_ras_path, b_ras_path, vec_path, mask_path, datasetOutDir, classCfgPath,
                                       SplittingBlockSize, SplittingStrideSize, uniform_name, mkdir_p, temp_dir,
                                       color_text_path, youhua)
        show_list.append(rasterize_path)
    else:
        if checkBoxSB_2:
            if len(v_series_path) != len(a_series_path):
                print("image: {},label:{} not equal".format(len(a_series_path), len(v_series_path)))
                return
            if not useChangeCheck:
                idx = 0
                for image, label in zip(a_series_path, v_series_path):
                    print('processing {}/{} file {} ...'.format(idx + 1, len(a_series_path), image))
                    idx += 1
                    if "yml" in image or "yaml" in image:
                        from .utils.msic import get_path_from_yaml
                        A = get_path_from_yaml(image)
                        V = get_path_from_yaml(label)
                        if not len(A) == len(V):
                            print("image: {},label:{} not equal".format(len(A), len(V)))
                            return
                        for a, v in zip(A, V):
                            uniform_name_temp = uniform_name
                            if len(uniform_name) > 2:
                                uniform_name = uniform_name + str(os.urandom(3).hex())
                            rasterize_path = diag_spliting(a, None, v, mask_path, datasetOutDir, classCfgPath,
                                                           SplittingBlockSize, SplittingStrideSize, uniform_name,
                                                           mkdir_p,
                                                           temp_dir,
                                                           color_text_path, youhua)
                            show_list.append(rasterize_path)
                            uniform_name = uniform_name_temp
                    else:
                        uniform_name_temp = uniform_name
                        if len(uniform_name) > 2:
                            uniform_name = uniform_name + str(os.urandom(3).hex())
                        rasterize_path = diag_spliting(image, None, label, mask_path, datasetOutDir, classCfgPath,
                                                       SplittingBlockSize, SplittingStrideSize, uniform_name, mkdir_p,
                                                       temp_dir,
                                                       color_text_path, youhua)
                        show_list.append(rasterize_path)
                        uniform_name = uniform_name_temp

        else:
            idx = 0
            for image in a_series_path:
                print('processing {}/{} file {} ...'.format(idx + 1, len(a_series_path), image))
                idx += 1
                uniform_name_temp = uniform_name
                if len(uniform_name) > 2:
                    uniform_name = uniform_name + str(os.urandom(3).hex())
                rasterize_path = diag_spliting(image, None, None, mask_path, datasetOutDir, classCfgPath,
                                               SplittingBlockSize, SplittingStrideSize, uniform_name, mkdir_p, temp_dir,
                                               color_text_path, youhua)
                show_list.append(rasterize_path)
                uniform_name = uniform_name_temp

        if useChangeCheck and checkBoxSB_2:
            if len(b_series_path) != len(a_series_path):
                print("image: {},post-image:{} not equal".format(len(a_series_path), len(b_series_path)))
                return
            idx = 0
            for image, imageB, label in zip(a_series_path, b_series_path, v_series_path):
                print('processing {}/{} file {} ...'.format(idx + 1, len(a_series_path), image))
                idx += 1
                if "yml" in image or "yaml" in image:
                    from .utils.msic import get_path_from_yaml
                    A = get_path_from_yaml(image)
                    B = get_path_from_yaml(imageB)
                    V = get_path_from_yaml(label)
                    if not len(A) == len(V) == len(B):
                        print("image: {},label:{},post image:{} not equal".format(len(A), len(V), len(B)))
                        return
                    for a, b, v in zip(A, B, V):
                        uniform_name_temp = uniform_name
                        if len(uniform_name) > 2:
                            uniform_name = uniform_name + str(os.urandom(3).hex())
                        rasterize_path = diag_spliting(a, b, v, mask_path, datasetOutDir, classCfgPath,
                                                       SplittingBlockSize, SplittingStrideSize, uniform_name, mkdir_p,
                                                       temp_dir,
                                                       color_text_path, youhua)
                        show_list.append(rasterize_path)
                        uniform_name = uniform_name_temp
                else:
                    uniform_name_temp = uniform_name
                    if len(uniform_name) > 2:
                        uniform_name = uniform_name + str(os.urandom(3).hex())
                    rasterize_path = diag_spliting(image, imageB, label, mask_path, datasetOutDir, classCfgPath,
                                                   SplittingBlockSize, SplittingStrideSize, uniform_name, mkdir_p,
                                                   temp_dir,
                                                   color_text_path, youhua)
                    show_list.append(rasterize_path)
                    uniform_name = uniform_name_temp
    from .utils.msic import del_dir
    del_dir(os.path.join(datasetOutDir))
    return show_list


class SplittingCreator(QObject):
    """A worker that posts a 'create splitting'"""
    fetched = pyqtSignal(list)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
            self,
            params: Dict[str, Any]
    ) -> None:
        """Initialize the worker.
        """
        super().__init__()
        self.params = params

    def create_splitting(self):
        """Initiate a processing."""
        if self.thread().isInterruptionRequested():
            self.finished.emit()
            return
        try:
            show_list = create_splitting(
                **self.params)  # [ rasterize_path = {"binary": "","muticlass": "",}
            self.fetched.emit(show_list)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


def mkdir_p(path):
    if not os.path.exists(path):
        os.makedirs(path)


def single_dection(mask_path: str, rasterComboA: List, rasterComboB: Union[List, None], useChangeDect: bool, cfg: Dict,
                   target_root: str, save_name: List):
    show_path = []
    # clip mask
    temp_dir = tempfile.gettempdir()
    if mask_path is not None:  # if b_ras_path vec_path classCfgPath mask_path
        if not os.path.exists(mask_path):
            print("mask layer doesn't exist")
            return
        else:
            rasterComboA = [clip_raster_by_mask(temp_dir, rasA, mkdir_p, mask_path) for rasA in rasterComboA]

            rasterComboB = [clip_raster_by_mask(temp_dir, rasB, mkdir_p,
                                                mask_path) for rasB in rasterComboB] if useChangeDect else None

    cfg.PNAME = [os.path.join(target_root, sa_name) for sa_name in save_name]
    cfg.PRE_IMG = rasterComboA
    cfg.POST_IMG = rasterComboB if useChangeDect else None

    if os.path.exists(cfg.USE_EXE):
        print('use exe in path :{}'.format(cfg.USE_EXE))
        exe_dir = os.path.dirname(cfg.USE_EXE)
        import sys
        sys.path.append(exe_dir) if exe_dir not in sys.path else None
        temp_configfile = os.path.join(target_root, 'config.yaml')
        # OmegaConf.to_yaml(cfg)
        print(cfg)
        OmegaConf.save(cfg, temp_configfile)
        command1 = "{} config_file={}".format(cfg.USE_EXE, temp_configfile)
        os.system(command1)
    else:
        cfg = OmegaConf.to_object(cfg)
        from .utils.inference_changedetection import full_flow
        full_flow(cfg)

    for name in save_name:
        show_path.append(
            dict(shp_path=os.path.join(target_root, name + '.shp'), png_path=os.path.join(target_root, name + '.png')))
    return show_path


def create_detection(
        useChangeDect, CfgPathName, processing_name, modelPathName, rasterComboA, rasterComboB, targetDirName,
        checkSHPBox, checkPNGBox, checkHALFBox, STRIDE_SIZE, BLOCKSIZE, BATCHSIZE, USECUDA, WEIGHT, maxval,
        simplify_ratiao, mask_path, useBatchCheck, a_series_path, b_series_path, use_exe
):
    show_lists = []
    cfg = OmegaConf.create(InferConfig)
    config_fn = CfgPathName
    if os.path.isfile(config_fn):
        user_conf = OmegaConf.load(config_fn)
        cfg = OmegaConf.merge(cfg, user_conf)
    else:
        cfg.model = None
    cfg.MODEL_PATH = modelPathName

    cfg.SAVE_SHP = checkSHPBox
    cfg.USE_EXE = use_exe
    cfg.SAVE_PNG = checkPNGBox
    cfg.HalfINFER = checkHALFBox
    cfg.STRIDE_SIZE = int(STRIDE_SIZE)
    cfg.BLOCKSIZE = int(BLOCKSIZE)
    cfg.BATCHSIZE = int(BATCHSIZE)
    cfg.USECUDA = USECUDA
    cfg.WEIGHT = WEIGHT
    cfg.maxval = int(maxval)
    cfg.simplify_ratiao = float(simplify_ratiao)
    target_root = targetDirName
    save_name = processing_name

    if not useBatchCheck:
        rasterComboB = [rasterComboB] if useChangeDect else None
        show_list = single_dection(mask_path, [rasterComboA], rasterComboB, useChangeDect, cfg, target_root,
                                   [save_name + '_' + os.path.splitext(os.path.basename(rasterComboA))[0]]
                                   )  # useChangeDect 控制输入
        show_lists.extend(show_list)
    else:
        temp_list_a = []
        temp_list_b = []
        for image, imageB in zip(a_series_path, b_series_path):
            if "yml" in image or "yaml" in image:
                from .utils.msic import get_path_from_yaml
                A = get_path_from_yaml(image)
                B = get_path_from_yaml(imageB) if useChangeDect else None
                show_list = single_dection(mask_path, A, B, useChangeDect, cfg,
                                           target_root,
                                           [save_name + '_' + os.path.splitext(os.path.basename(rasA))[0] for rasA in A]
                                           )
                show_lists.extend(show_list)
            else:
                temp_list_a.append(image)
                temp_list_b.append(imageB)
        temp_list_b = temp_list_b if useChangeDect else None
        show_list = single_dection(mask_path, temp_list_a, temp_list_b, useChangeDect, cfg,
                                   target_root,
                                   [save_name + '_' + os.path.splitext(os.path.basename(rasA))[0] for rasA in
                                    temp_list_a]
                                   )
        show_lists.extend(show_list)
    return show_lists


class DetectionCreator(QObject):
    """A worker that posts a 'create Detection'"""
    fetched = pyqtSignal(list)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
            self,
            params: Dict[str, Any]
    ) -> None:
        """Initialize the worker.
        """
        super().__init__()
        self.params = params

    def create_detection(self):
        """Initiate a processing."""
        if self.thread().isInterruptionRequested():
            self.finished.emit()
            return
        try:
            show_list = create_detection(**self.params)
            self.fetched.emit(show_list)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
