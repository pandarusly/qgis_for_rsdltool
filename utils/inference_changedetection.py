import os
import sys

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from osgeo import gdal, ogr, osr
from torch.utils.data import DataLoader

from .msic import (ImageSlicer, TileMerger, get_time, image_to_tensor,
                   to_2tuple, to_numpy)


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
    # width, height = band.XSize, band.YSize
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
    # 创建结果Geojson 
    json_driver = ogr.GetDriverByName('GeoJSON')
    json_ds = json_driver.CreateDataSource(dst_shpfile.replace('.shp','.json'))
   
    json_lyr = json_ds.CreateLayer('Result', layer.GetSpatialRef()) 
    # -------------
    fieldname = ogr.FieldDefn("class_name", ogr.OFTString)
    json_lyr.CreateField(fieldname)
    fieldname = ogr.FieldDefn("segment", ogr.OFTInteger)
    json_lyr.CreateField(fieldname)
    # -------------  
    for feature in layer: 
        geom = feature.GetGeometryRef() 
        json_feat = ogr.Feature(json_lyr.GetLayerDefn())
        json_feat.SetGeometry(geom)  
 
        if feature.GetField('segment') == 1:
            json_feat.SetField("class_name", "建筑物变化")
            json_feat.SetField("segment", 1)
        elif feature.GetField('segment') == 2:
            json_feat.SetField("class_name", "推填土变化")
            json_feat.SetField("segment", 2)
        elif feature.GetField('segment') == 3:
            json_feat.SetField("class_name", "道路变化")
            json_feat.SetField("segment", 3)
 
        json_lyr.CreateFeature(json_feat) 
        
    json_ds.Destroy()

    datasource.Destroy()



    
    print("---------complete polygonized!------------------")



@get_time("polygonizedAndGeojson")
def polygonizedAndGeojson(src_ds, dst_shpfile, *args, **kwargs):
    print("---------start polygonizedAndGeojson!------------------")
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES") 
    gdal.SetConfigOption("SHAPE_ENCODING", "GBK")
    ogr.RegisterAll()

    shp_dr = ogr.GetDriverByName("ESRI Shapefile")

    band = src_ds.GetRasterBand(1)  # 0，1 
    srs = osr.SpatialReference()
    srs.ImportFromWkt(src_ds.GetProjectionRef())
 
    shp_ds = shp_dr.CreateDataSource(dst_shpfile)
    shp_lyr = shp_ds.CreateLayer('Result', srs, ogr.wkbPolygon)
    fieldDefn = ogr.FieldDefn('segment', ogr.OFTInteger)
    shp_lyr.CreateField(fieldDefn, 1)
    gdal.Polygonize(band, band, shp_lyr, 0)
 

    # 创建结果Geojson 
    json_driver = ogr.GetDriverByName('GeoJSON')
    json_ds = json_driver.CreateDataSource(dst_shpfile.replace('.shp','.json'))
   
    json_lyr = json_ds.CreateLayer('Result', shp_lyr.GetSpatialRef()) 
    # -------------
    fieldname = ogr.FieldDefn("class_name", ogr.OFTString)
    json_lyr.CreateField(fieldname)
    fieldname = ogr.FieldDefn("segment", ogr.OFTInteger)
    json_lyr.CreateField(fieldname)
    # -------------
      
    for feature in shp_lyr: 
        geom = feature.GetGeometryRef() 
        json_feat = ogr.Feature(json_lyr.GetLayerDefn())
        json_feat.SetGeometry(geom)  
 
        if feature.GetField('segment') == 1:
            json_feat.SetField("class_name", "建筑物变化")
            json_feat.SetField("segment", 1)
        elif feature.GetField('segment') == 2:
            json_feat.SetField("class_name", "推填土变化")
            json_feat.SetField("segment", 2)
        elif feature.GetField('segment') == 3:
            json_feat.SetField("class_name", "道路变化")
            json_feat.SetField("segment", 3)
 
        json_lyr.CreateFeature(json_feat) 
      

    json_ds.Destroy()
    shp_ds.Destroy()


    # ---删除 shp附属文件
    # os.remove(dst_shpfile)
    # os.remove(dst_shpfile.replace('.shp','.shx'))
    # os.remove(dst_shpfile.replace('.shp','.prj'))
    # os.remove(dst_shpfile.replace('.shp','.dbf'))
    print("---------complete polygonized!------------------")
    return dst_shpfile.replace('.shp','.json')
 
@get_time("load model")
def load_jit_model(MODEL_PATH, device, MODEL_CFG=None, half_infer=False):
    print("---------start load_model!------------------")
    print(MODEL_PATH)
    if MODEL_CFG is not None:
        try:
            import hydra
        except ImportError:
            print("hydra is not installed")
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname((abspath))
        dname = os.path.dirname((dname))
        sys.path.append(dname)
        print(dname)
        # MODEL_CFG['Config']['MODEL']['pretrained'] = None
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


@get_time("load model")
def load_model(MODEL_CFG, MODEL_PATH, device, half_infer=False):
    try:
        import hydra
    except ImportError:
        print("hydra is not installed")
    print("---------start load_model!------------------")
    config = OmegaConf.load(MODEL_CFG)
    config.model.Config.MODEL.pretrained = None
    model = hydra.utils.instantiate(config.model, _convert_="all")
    model.load_state_dict(torch.load(
        MODEL_PATH, map_location=device)["state_dict"])
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
        if tiles_batch.shape[1] == 2:
            seg_score = model(tiles_batch[:, 0], tiles_batch[:, 1])
        else:
            seg_score = model(tiles_batch)

        # predict label
        if isinstance(seg_score, list):
            ort_outs = seg_score[0].argmax(1)
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
    tiler, tiles1, width, height, source_config = get_tiles(
        RSImage, blockSize, strideSize, weight)
    if os.path.exists(RSImage_another):
        _, tiles2, _, _, _ = get_tiles(
            RSImage_another, blockSize, strideSize, weight)
        # tiles = [torch.stack([t1, t2], dim=0)
        #          for t1, t2 in zip(tiles1, tiles2)]
        tiles = [torch.cat([t1, t2], dim=0)
                 for t1, t2 in zip(tiles1, tiles2)]
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
        ort_outs = batch_forward(
            model=model, tiles_batch=tiles_batch, device=device, half_infer=half_infer)
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


def cv2_post_dealv2(img, class_nums=4):
    """这里假设预测结果已经保存在名为 prediction.png 的文件中，且分为了背景、物体1、物体2 和物体3 四类。我们首先使用中值滤波去除噪声，然后使用 OpenCV 提供的 connectedComponentsWithStats 函数进行连通性检测，将不同物体之间的分割线断开。
    """
    h, w = img.shape[0], img.shape[1] 
    # 去噪
    # img = cv2.medianBlur(img, 5)
    # 定义开运算结构元素，可以根据实际情况调整大小
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    kernel = np.ones((3, 3), np.uint8)
    # 遍历每个类别的标签图进行处理
    for i in range(1, class_nums):
        # 创建一个只包含当前类别的掩膜图像
        if i==2:
            mask = np.uint8(img == i) * 255 
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for j in range(len(contours)):
                area = cv2.contourArea(contours[j])
                if area > 500:  # 设定筛选阈值，可根据实际情况调整
                    cv2.drawContours(mask, [contours[j]], -1, 0, -1)
            # 对掩膜图像执行开运算操作
            # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            # mask = cv2.dilate(mask, kernel)  # 膨胀
            # mask = cv2.erode(mask, kernel)   # 腐蚀
            # 将处理后的掩膜图像重新合并到原始图像中
            img = np.where(mask == 255, 3, img)


    return img

@get_time('total')
def full_flow(cfg):

    OmegaConf.save(cfg,os.path.join(os.path.dirname(cfg["PNAME"][0]),"config.yaml"))
    MODEL_PATH = cfg["MODEL_PATH"]
    # MODEL_CFG = cfg["MODEL_CFG"]
    MODEL_CFG = cfg["model"]
    RSImage = cfg["PRE_IMG"]
    RSImage_another = cfg["POST_IMG"]

    assert isinstance(RSImage, (list, tuple)
                      ), "RSImage : {} Not List".format(RSImage)
    # assert isinstance(RSImage_another, (list, tuple)), "RSImage_another : {} Not List".format(RSImage)
    assert isinstance(cfg["PNAME"], (list, tuple)
                      ), "PNAME : {} Not List".format(cfg["PNAME"])

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

    model, device = load_jit_model(
        MODEL_PATH, device, MODEL_CFG, half_infer=half_infer)
    print("use batch inference!!!\n")
    # print("{},{}!!!\n".format(RSImage, RSImage_another))

    if RSImage_another is None:
        RSImage_another = ["" for ra in RSImage]

    for RSImage_, RSImage_another_, PNAME in zip(RSImage, RSImage_another, cfg["PNAME"]):
        # print(RSImage_another_)
        mem_ds = gdal_cd_inference(RSImage=RSImage_, RSImage_another=RSImage_another_,
                                   model=model, blockSize=blockSize, strideSize=strideSize, device=device,
                                   batch_size=batch_size, weight=weight, half_infer=half_infer)

        BASENAME = os.path.basename(PNAME).split('.')[0]
        DIRNAME = os.path.dirname(PNAME)
        PNGPATH = os.path.join(DIRNAME, BASENAME + '.tif')
        SHPPATH = os.path.join(DIRNAME, BASENAME)
        dst_shpfile = SHPPATH + '.shp'
        if not os.path.exists(DIRNAME):
            os.makedirs(DIRNAME)
        # # # ----------------------------------- 保存普通图片
        if cfg["SAVE_PNG"]:
            print("---------start save png!-----------------")
            ct = gdal.ColorTable()
            NLCD_COLORMAP = {
                0: (0, 0, 0, 255),
                1: (31, 119, 180, 255),
                2: (174, 199, 232, 255),
                3: (255, 127, 14, 255),
            }
            for key in NLCD_COLORMAP.keys():
                ct.SetColorEntry(key, NLCD_COLORMAP[key])
            mem_ds.GetRasterBand(1).SetRasterColorTable(ct) 
            # save_to_PNG(mem_ds,PNGPATH)
            merged = mem_ds.ReadAsArray()
            merged = cv2_post_dealv2(merged,class_nums=4)
            mem_ds.GetRasterBand(1).WriteArray(merged)
            # drv = gdal.GetDriverByName("PNG")
            drv = gdal.GetDriverByName("GTiff")
            dst_ds = drv.CreateCopy(PNGPATH, mem_ds)
            # dst_ds = drv.CreateCopy(PNGPATH.replace('png','tif'), mem_ds)
            dst_ds = None
            # cv2.imwrite(PNGPATH, merged)
            # cv2.imencode('.png', merged)[1].tofile(PNGPATH)
            print("---------save png in {}----------------\n".format(PNGPATH))
        if cfg["SAVE_SHP"]:
            # ----------------------------- polygonized
            simp_dst_shpfile = SHPPATH + '_simplyfie.shp'
            polygonized(mem_ds, dst_shpfile, simp_dst_shpfile,
                        simplify_ratiao=cfg['simplify_ratiao'])
            # json_file = polygonizedAndGeojson(mem_ds,PNGPATH.replace('.png','.shp'))
            # with open(json_file, 'r', encoding='utf-8') as f: 
            #     content = json.load(f)
            # for id, feature in enumerate(content['features']): #extract the geometry from the feature 
            #     geometry = feature['geometry'] #create a shapely object from the geometry 
            #     shape = shapely.geometry.shape(geometry) #simplify the shape with a tolerance of 0.001
            #     simplified_shape = shape.simplify(tolerance=0.0001) #add the simplified shape to the empty list 
            #     # simplified_data.append(simplified_shape) 
            #     simplified_shape = shapely.to_geojson(simplified_shape) # str -> dict
            #     simplified_shape=json.loads(simplified_shape)
            #     content['features'][id]['geometry']  = simplified_shape
            # with open(simp_dst_shpfile.replace(".shp",".json"), 'w') as f:
            #     json.dump(content, f)  # 编码JSON数据 

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
            raise FileNotFoundError(
                f"config_file={config_fn} is not a valid file")

    cfg = OmegaConf.to_object(OmegaConf.merge(  # Merge in any arguments passed via the command line
        cfg, command_line_conf
    ))

    print(OmegaConf.to_yaml(cfg))

    full_flow(cfg)
