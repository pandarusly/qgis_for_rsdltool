import cv2
import numpy as np

try:
    from osgeo import gdal
except:
    import gdal

import os.path as osp
import math


def ImageRowCol2Projection(adfGeoTransform, iCol, iRow):
    """translate from image row and col to projection"""
    dProjX = adfGeoTransform[0] + adfGeoTransform[1] * iCol + adfGeoTransform[2] * iRow
    dProjY = adfGeoTransform[3] + adfGeoTransform[4] * iCol + adfGeoTransform[5] * iRow
    return dProjX, dProjY


def GetGeoExtent(x_off, y_off, cropsize, adfGeoTransform):
    """get batch extent"""
    startX, startY = ImageRowCol2Projection(adfGeoTransform, x_off, y_off)
    endX, endY = ImageRowCol2Projection(adfGeoTransform, x_off + cropsize, y_off + cropsize)
    extent = [startX, endX, endY, startY]
    return extent


def is_label_useful(array, threshold=500, len_contours=0, **kwargs) -> bool:
    _, thresh = cv2.threshold(array, thresh=1.5, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 统一变成255前景
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 寻找连通域
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(contours[i])
        if area < threshold:
            cv2.drawContours(thresh, [contours[i]], -1, 0, -1)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 填充后寻找连通域
    if len(contours) > len_contours:
        return True
    return False


def is_img_useful(array, width, height, valpixel: float = 0.2, **kwargs) -> bool:
    array[array > 5] = 1
    valpixelNum = float(array.sum())
    if valpixelNum / (width * height) < valpixel:
        return False
    return True


def ExportTile(ds, cdpath, file_name, startX, startY, x_off, y_off, gt, cropsize, imgfrmat, frmt_ext, i, j,
               youhua):
    # 处理标签有bug  bufer不够  scanline
    # """export one image batch sample"""
    adfGeoNew = (startX, gt[1], gt[2], startY, gt[4], gt[5])
    driver = gdal.GetDriverByName("MEM")
    channels = 3 if imgfrmat == "JPEG" else 1
    mem_ds = driver.Create("", cropsize, cropsize, channels, gdal.GDT_Byte)
    scanline = ds.ReadRaster(x_off, y_off, cropsize, cropsize, cropsize, cropsize, gdal.GDT_Byte)
    mem_ds.WriteRaster(0, 0, cropsize, cropsize, scanline, cropsize, cropsize, gdal.GDT_Byte)
    array = mem_ds.GetRasterBand(1).ReadAsArray()
    width, height = mem_ds.GetRasterBand(1).XSize, mem_ds.GetRasterBand(1).YSize
    if youhua is not None:
        if imgfrmat == "JPEG":  # 刪除掉有用像素占比較少的
            if not is_img_useful(array, width, height, **youhua):
                dst_ds = None
                mem_ds = None
                return None
        else:
            if not is_label_useful(array, **youhua):
                dst_ds = None
                mem_ds = None
                return None
    mem_ds.SetGeoTransform(adfGeoNew)
    mem_ds.SetProjection(ds.GetProjection())
    if imgfrmat != "JPEG":
        array = array * 255 if np.max(array) < 255 else array
        mem_ds.GetRasterBand(1).WriteArray(array)

    drv = gdal.GetDriverByName(imgfrmat)
    imagepath = osp.join(cdpath, (str(file_name) + "-" + str(j) + "-" + str(i) + "." + frmt_ext))
    dst_ds = drv.CreateCopy(imagepath, mem_ds)
    dst_ds = None
    mem_ds = None
    return 'successful'


# Start Splitting
def splitting(fn_ras, cdpath, frmt_ext, imgfrmat, scaleoptions, cropsize, stridesize, file_name, youhua=dict(valpixel=0.2, threshold=500, len_contours=0)):
    ds = gdal.Open(fn_ras)
    gt = ds.GetGeoTransform()
    # get coordinates of upper left corner
    resx = gt[1]
    res_y = gt[5]
    resy = abs(res_y)

    width, height = ds.RasterXSize, ds.RasterYSize

    rows_max = int((height - cropsize) / stridesize)
    columns_max = int((width - cropsize) / stridesize)  # 加1则最后一列返回一个block

    idx = 0
    for i in range(rows_max + 1):
        for j in range(columns_max + 1):
            # print('processing {}/{} file ...'.format(idx + 1, columns_max * rows_max))
            idx += 1
            x_off = width - cropsize if j == columns_max else j * stridesize
            y_off = height - cropsize if i == rows_max else i * stridesize

            startX, endX, endY, startY = GetGeoExtent(x_off, y_off, cropsize, gt)
            ExportTile(ds, cdpath, file_name, startX, startY, x_off, y_off, gt, cropsize, imgfrmat, frmt_ext, i, j,
                       youhua)
            # or gdal translate to subset the input raster
            # if len(str(ds.GetProjection())) < 1:
            #     # ExportImgae(ds, cdpath, file_name, startX, startY, x_off, y_off, gt, cropsize, imgfrmat, frmt_ext, i, j)
            #     gdal.Translate(osp.join(cdpath, (str(file_name) + "-" + str(j) + "-" + str(i) + "." + frmt_ext)),
            #                    ds,
            #                    projWin=(abs(startX), abs(startY), abs(endX), abs(endY)),
            #                    outputType=gdal.gdalconst.GDT_Byte,
            #                    format=imgfrmat,
            #                    scaleParams=[[scaleoptions]])
            # else:
            #     #
            #     gdal.Translate(osp.join(cdpath, (str(file_name) + "-" + str(j) + "-" + str(i) + "." + frmt_ext)),
            #                    ds,
            #                    projWin=(abs(startX), abs(startY), abs(endX), abs(endY)),
            #                    xRes=resx,
            #                    yRes=-resy,
            #                    outputType=gdal.gdalconst.GDT_Byte,
            #                    format=imgfrmat,
            #                    scaleParams=[[scaleoptions]])

    # close the open dataset!!!
    ds = None
