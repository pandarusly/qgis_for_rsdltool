import os
import sys

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from torch.utils.data import DataLoader

from .msic import (to_numpy, TileMerger, ImageSlicer, image_to_tensor, get_time, to_2tuple)


@get_time("cv2_post_deal")
def cv2_post_deal(img, maxval=255, PolyDP=4):
    h, w = img.shape[0], img.shape[1]
    pre_image = np.zeros(shape=(h, w), dtype=np.uint8)
    _, thresh = cv2.threshold(
        img, thresh=1.5, maxval=maxval, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, 3, 2)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
        approx1 = cv2.approxPolyDP(cnt, PolyDP, True)
        # 万能操作 集合上个两个API优点 填充＋绘制
        cv2.drawContours(pre_image, [approx1], -1, (maxval, 0, 0), -1)
    return pre_image


@get_time("polygonized")
def polygonized(src_ds, dst_shpfile, simp_dst_shpfile, simplify_ratiao=1):
    print("---------start polygonized!------------------")
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    gdal.SetConfigOption("SHAPE_ENCODING", "CP936")
    ogr.RegisterAll()

    shp_dr = ogr.GetDriverByName("ESRI Shapefile")

    band = src_ds.GetRasterBand(1)  # 0，1
    width, height = band.XSize, band.YSize
    srs = osr.SpatialReference()
    srs.ImportFromWkt(src_ds.GetProjectionRef())

    if not os.path.exists(os.path.dirname(dst_shpfile)):
        os.makedirs(os.path.dirname(dst_shpfile))

    datasource = shp_dr.CreateDataSource(dst_shpfile)
    layer = datasource.CreateLayer('Result', srs, ogr.wkbPolygon)
    fieldDefn = ogr.FieldDefn('segment', ogr.OFTInteger)
    layer.CreateField(fieldDefn, 1)
    gdal.Polygonize(band, band, layer, 0)

    # if simplify_ratiao >0:
    #     for in_feat in layer:
    #         geom = in_feat.GetGeometryRef()
    #         simplify_geom = geom.Simplify(simplify_ratiao).ExportToWkb()
    #         polygon = ogr.CreateGeometryFromWkb(simplify_geom)
    #         # print(simplify_geom)
    #         in_feat.SetGeometry(polygon)
    #         layer.SetFeature(in_feat)
    datasource.Destroy()
    print("---------complete polygonized!------------------")


@get_time("load model")
def load_jit_model(MODEL_PATH, device, MODEL_CFG=None, half_infer=False):
    print("---------start load_model!------------------")
    print(MODEL_PATH)
    if MODEL_CFG is not None:
        import hydra
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname((abspath))
        dname = os.path.dirname((dname))
        sys.path.append(dname)
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


@get_time("load model")
def load_model(MODEL_CFG, MODEL_PATH, device, half_infer=False):
    import hydra
    print("---------start load_model!------------------")
    config = OmegaConf.load(MODEL_CFG)
    config.model.Config.MODEL.pretrained = None
    model = hydra.utils.instantiate(config.model, _convert_="all")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device)["state_dict"])
    if half_infer:
        model.half()
    model.to(device)
    model.eval()

    print("---------complete load_model!------------------")

    return model, device


def batch_forward(model, tiles_batch, device, half_infer=False):
    tiles_batch = ((tiles_batch.to(torch.float32) / 255.0) - 0.5) / 0.5
    tiles_batch = tiles_batch.to(device)
    if half_infer:
        tiles_batch = tiles_batch.half()

    with torch.no_grad():
        seg_score = model(tiles_batch[:, 0], tiles_batch[:, 1])
        # predict label
        if isinstance(seg_score, list):
            ort_outs = seg_score[-1].argmax(1)
        else:
            ort_outs = seg_score.argmax(1)
        ort_outs = ort_outs.unsqueeze(1)

    return ort_outs


def get_tiles(RSImagePath, blockSize, strideSize, weight):
    rsImg_ds_another = gdal.Open(RSImagePath, gdal.GA_ReadOnly)
    if rsImg_ds_another is None:
        print("open raster data:{} failed!".format(RSImagePath))
        sys.exit(1)
    band2 = rsImg_ds_another.GetRasterBand(1)
    width2, height2 = band2.XSize, band2.YSize

    image2 = rsImg_ds_another.ReadAsArray(0, 0, width2, height2)
    # image2 = image2[::-1, :, :] bgr - > rgb
    image2 = image2[-3:, ...]
    image2 = np.moveaxis(image2, 0, -1)
    tiler_temp = ImageSlicer(image2.shape, tile_size=blockSize,
                             tile_step=strideSize, weight=weight)  # pyramid
    tiles2 = [image_to_tensor(tile) for tile in tiler_temp.split(image2)]

    proj = rsImg_ds_another.GetProjectionRef()
    transform = rsImg_ds_another.GetGeoTransform()
    rsImg_ds_another = None
    return tiler_temp, tiles2, width2, height2, (proj, transform)


@get_time("inference")
def gdal_cd_inference(RSImage, RSImage_another, model,
                      blockSize=(256, 256),
                      strideSize=(256, 256),
                      device="cuda",
                      batch_size=4,
                      weight="mean",
                      half_infer=False):
    print("---------start inference!------------------")
    # open the rs data with gdal

    # ##############################
    # # inference block by block   #
    # ##############################
    tiler, tiles1, width, height, source_config = get_tiles(RSImage, blockSize, strideSize, weight)
    if os.path.exists(RSImage_another):
        _, tiles2, _, _, _ = get_tiles(RSImage_another, blockSize, strideSize, weight)
        tiles = [torch.stack([t1, t2], dim=0) for t1, t2 in zip(tiles1, tiles2)]
        datamodule = DataLoader(list(zip(tiles, tiler.crops)),
                                batch_size=batch_size, pin_memory=True)
    else:
        datamodule = DataLoader(list(zip(tiles1, tiler.crops)),
                                batch_size=batch_size, pin_memory=True)

    nums = len(datamodule)
    idx = 0
    merger = TileMerger(tiler.target_shape, 1, tiler.weight, device=device)
    for tiles_batch, coords_batch in datamodule:
        print('processing {}/{} batchs ...'.format(idx + 1, nums))
        ort_outs = batch_forward(model=model, tiles_batch=tiles_batch, device=device, half_infer=half_infer)
        merger.integrate_batch(ort_outs, coords_batch)
        idx += 1
    # -------------------------- 合并
    merged = np.moveaxis(to_numpy(merger.merge()), 0, -1).astype(np.uint8)
    merged = tiler.crop_to_orignal_size(merged)
    # --------------设置缓存接受输出
    mem_dr = gdal.GetDriverByName("MEM")
    mem_ds = mem_dr.Create("", width, height, 1, gdal.GDT_Byte)
    mem_ds.GetRasterBand(1).WriteArray(merged[:, :, 0])  # 直接收 w,h
    srs = osr.SpatialReference()
    srs.ImportFromWkt(source_config[0])
    mem_ds.SetProjection(srs.ExportToWkt())
    mem_ds.SetGeoTransform(source_config[1])

    print("---------complete inference!------------------")

    return mem_ds


@get_time('total')
def full_flow(cfg):
    MODEL_PATH = cfg["MODEL_PATH"]
    # MODEL_CFG = cfg["MODEL_CFG"]
    MODEL_CFG = cfg["model"]
    RSImage = cfg["PRE_IMG"]
    RSImage_another = cfg["POST_IMG"]

    assert isinstance(RSImage, (list, tuple)), "RSImage : {} Not List".format(RSImage)
    # assert isinstance(RSImage_another, (list, tuple)), "RSImage_another : {} Not List".format(RSImage)
    assert isinstance(cfg["PNAME"], (list, tuple)), "PNAME : {} Not List".format(cfg["PNAME"])

    # print("pre_img :{}\n".format(RSImage))
    # print("post_img :{}\n".format(RSImage_another))
    blockSize = cfg["BLOCKSIZE"]
    strideSize = cfg["STRIDE_SIZE"]
    batch_size = cfg["BATCHSIZE"]
    weight = cfg["WEIGHT"]
    half_infer = cfg["HalfINFER"]
    if half_infer:
        torch._C._jit_override_can_fuse_on_gpu(False)

    if not isinstance(blockSize, (list, tuple)):
        blockSize = to_2tuple(blockSize)

    if not isinstance(strideSize, (list, tuple)):
        strideSize = to_2tuple(strideSize)

    print("torch.cuda.is_availabl:{} \n".format(torch.cuda.is_available()))
    if cfg["USECUDA"]:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    print("device:{}\n".format(device))

    model, device = load_jit_model(MODEL_PATH, device, MODEL_CFG, half_infer=half_infer)
    print("use batch inference!!!\n")

    if RSImage_another is None:
        RSImage_another = ["" for ra in RSImage]

    for RSImage_, RSImage_another_, PNAME in zip(RSImage, RSImage_another, cfg["PNAME"]):
        print(RSImage_another_)
        mem_ds = gdal_cd_inference(RSImage=RSImage_, RSImage_another=RSImage_another_,
                                   model=model, blockSize=blockSize, strideSize=strideSize, device=device,
                                   batch_size=batch_size, weight=weight, half_infer=half_infer)
        import os
        BASENAME = os.path.basename(PNAME).split('.')[0]
        DIRNAME = os.path.dirname(PNAME)
        PNGPATH = os.path.join(DIRNAME, BASENAME + '.png')
        SHPPATH = os.path.join(DIRNAME, BASENAME)
        dst_shpfile = SHPPATH + '.shp'
        if not os.path.exists(DIRNAME):
            os.makedirs(DIRNAME)
        # # # ----------------------------------- 保存普通图片
        if cfg["SAVE_PNG"]:
            print("---------start save png!-----------------")
            merged = mem_ds.ReadAsArray()
            merged = cv2_post_deal(merged, maxval=cfg['maxval'], PolyDP=cfg['PolyDP'])
            # mem_ds.GetRasterBand(1).WriteArray(merged)
            # drv = gdal.GetDriverByName("PNG")
            # dst_ds = drv.CreateCopy(PNGPATH, mem_ds)
            # dst_ds = None
            # cv2.imwrite(PNGPATH, merged)
            cv2.imencode('.png', merged)[1].tofile(PNGPATH)
            print("---------save png in {}----------------\n".format(PNGPATH))
        if cfg["SAVE_SHP"]:
            # ----------------------------- polygonized
            simp_dst_shpfile = SHPPATH + '_simplyfie.shp'
            polygonized(mem_ds, dst_shpfile, simp_dst_shpfile, simplify_ratiao=cfg['simplify_ratiao'])


if __name__ == '__main__':

    # imgs = ['test_102_0512_0000', 'test_121_0768_0256', 'test_2_0000_0512', 'test_77_0512_0256', 'train_36_0512_0512',
    #         'train_412_0512_0768']

    command_line_conf = OmegaConf.from_cli()
    cfg = OmegaConf.create(
        dict(
            MODEL_PATH=r"huanjing_mixuper.pt",
            # MODEL_PATH=r"D:\深度学习\代码\变化检测和语义分割代码\mmlightv2\logs\experiments\runs\bye\mit_uper_levir1024\2022-06-06_14-34-16\checkpoints\epoch_103.ckpt",
            PRE_IMG=r"D:\zhuomian\data\qd_data\qd201806.img",
            POST_IMG=r"D:\zhuomian\data\qd_data\qd201807.img",
            HalfINFER=True,
            STRIDE_SIZE=512,
            BLOCKSIZE=512,
            BATCHSIZE=64,
            USECUDA=True,
            MUTI=True,
            WEIGHT='pyramid',  # mean
            PNAME=r'data/predict_huanjing/qd201807',
            SAVE_SHP=True,
            SAVE_PNG=True,
            maxval=255,
            PolyDP=1,
            simplify_ratiao=1,
            MODEL_CFG=r"D:\深度学习\代码\变化检测和语义分割代码\mmlightv2\logs\experiments\runs\bye\mit_uper_levir1024\2022-06-06_14-34-16\.hydra\config.yaml"
        )
    )

    if "config_file" in command_line_conf:
        config_fn = command_line_conf.config_file
        if os.path.isfile(config_fn):
            user_conf = OmegaConf.load(config_fn)
            conf = OmegaConf.merge(cfg, user_conf)
        else:
            raise FileNotFoundError(f"config_file={config_fn} is not a valid file")

    cfg = OmegaConf.to_object(OmegaConf.merge(  # Merge in any arguments passed via the command line
        cfg, command_line_conf
    ))

    print(OmegaConf.to_yaml(cfg))

    full_flow(cfg)
