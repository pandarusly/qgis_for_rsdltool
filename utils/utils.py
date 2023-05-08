import numpy as np

try:
    from osgeo import gdal
except:
    import gdal
from osgeo import ogr, osr

# gdal 的一些函数工具
def GetSpatialFilterLayer(x_off, y_off, cropsize, adfGeoTransform, srs, oLayer):
    """get the cliped layer,which include the max range of target"""
    startX, startY = ImageRowCol2Projection(adfGeoTransform, x_off, y_off)
    endX, endY = ImageRowCol2Projection(adfGeoTransform, x_off + cropsize, y_off + cropsize)
    wkt = "POLYGON((%s %s,%s %s,%s %s,%s %s,%s %s))" % (
        startX, startY, endX, startY, endX, endY, startX, endY, startX, startY)
    geomRectangle = ogr.CreateGeometryFromWkt(wkt)
    oLayer.SetSpatialFilterRect(startX, endY, endX, startY)
    driver = ogr.GetDriverByName("Memory")
    mem_ds = driver.CreateDataSource("")
    result_layer = mem_ds.CreateLayer("polygon", srs, ogr.wkbPolygon)
    oDefn = result_layer.GetLayerDefn()
    new_extent = None
    # caculate new extent
    for feat in oLayer:
        geom = feat.GetGeometryRef()
        oFeatureRectangle = ogr.Feature(oDefn)
        oFeatureRectangle.SetGeometry(geom)
        result_layer.CreateFeature(oFeatureRectangle)
    if oLayer.GetFeatureCount() > 0:
        ext = result_layer.GetExtent()
        wkt1 = "POLYGON((%s %s,%s %s,%s %s,%s %s,%s %s))" % (
            ext[0], ext[3], ext[1], ext[3], ext[1], ext[2], ext[0], ext[2], ext[0], ext[3])
        geom_new = ogr.CreateGeometryFromWkt(wkt1)
        temp = geom_new.Union(geomRectangle)
        new_extent = temp.GetEnvelope()
    oLayer.SetSpatialFilter(None)
    return new_extent


def GetGeoExtent(x_off, y_off, cropsize, adfGeoTransform):
    """get batch extent"""
    startX, startY = ImageRowCol2Projection(adfGeoTransform, x_off, y_off)
    endX, endY = ImageRowCol2Projection(adfGeoTransform, x_off + cropsize, y_off + cropsize)
    extent = [startX, endX, endY, startY]
    return extent


def GetInvalidFID(Layer, adfGeoTransform):
    """return layer that delete small area"""
    arean = -adfGeoTransform[1] * adfGeoTransform[5]
    count = 0
    pixels = 0
    for feat in Layer:
        geom = feat.GetGeometryRef()
        area = geom.GetArea()
        pixel = (int)(area / arean)
        pixels = pixels + pixel
        count = count + 1
    if count == 0:
        return Layer
    threshold = 0.2 * (pixels / count)
    for feat in Layer:
        fid = feat.GetFID()
        geom = feat.GetGeometryRef()
        area = geom.GetArea()
        if area <= threshold:
            Layer.DeleteFeature(fid)
    return Layer


def GetPercentMaxMin(dataset):
    """get percent strecth boundary"""
    scalelist = []
    width, height = dataset.GetRasterBand(1).XSize, dataset.GetRasterBand(1).YSize
    imgsize = width * height
    nbandcount = dataset.RasterCount
    strectch = 0.02
    for index in range(nbandcount):
        band = dataset.GetRasterBand(index + 1)
        stretchmin = 0
        stretchmax = 0
        ratio = 0
        histogram, buckets, minb, maxb = GetHistogram(band)
        for i in range(buckets):
            ratio = ratio + histogram[i]
            if stretchmin == 0 and ratio >= strectch:
                stretchmin = i + minb
            if stretchmax == 0 and ratio >= (1 - strectch):
                stretchmax = i + minb
            if i == buckets - 1 and ratio <= (1 - strectch):
                stretchmax = maxb
        scalelist.append([stretchmin, stretchmax, 0, 255])
    print(scalelist)
    return tuple(scalelist)


def GetHistogram(band):
    """Get array histogram"""
    width, height = band.XSize, band.YSize
    array = band.ReadAsArray(0, 0, width, height, width, height)
    minv = array.min()
    maxv = array.max()
    buckets = maxv - minv
    (n, bins) = np.histogram(array, bins=buckets, range=(minv + 1, maxv), normed=True)
    return n, buckets, minv, maxv


def GetTBLXList(layer, tblx_field):
    """Get unique TBLX list"""
    if tblx_field is None:
        return []
    TBLXList = []
    for feature in layer:
        tblx = feature.GetFieldAsString(tblx_field)
        if tblx not in TBLXList:
            TBLXList.append(tblx)
    return TBLXList


def ExportlabelTile(layer, adfgeotransform, proj, cropsize, labelpath):
    """shpfile to raster"""
    memdr = gdal.GetDriverByName("mem")
    memds = memdr.Create("", cropsize, cropsize, 1, gdal.GDT_Byte)
    memds.SetGeoTransform(adfgeotransform)
    memds.SetProjection(proj)
    gdal.RasterizeLayer(memds, [1], layer, burn_values=[255])
    dr = gdal.GetDriverByName("PNG")
    tempds = dr.CreateCopy(labelpath, memds)
    tempds = None
    memds = None


def ExportlabelTile_class(layer, adfgeotransform, proj, cropsize, labelpath, TBBM_field, NLCD_COLORMAP):
    """shpfile to raster with attribute"""
    memdr = gdal.GetDriverByName("mem")
    memds = memdr.Create("", cropsize, cropsize, 1, gdal.GDT_Byte)
    memds.SetGeoTransform(adfgeotransform)
    memds.SetProjection(proj)
    ct = gdal.ColorTable()
    for key in NLCD_COLORMAP.keys():
        ct.SetColorEntry(key, NLCD_COLORMAP[key])
    memds.GetRasterBand(1).SetRasterColorTable(ct)
    OPTIONS = 'ATTRIBUTE=' + TBBM_field
    gdal.RasterizeLayer(memds, [1], layer, options=[OPTIONS])
    dr = gdal.GetDriverByName("PNG")
    tempds = dr.CreateCopy(labelpath, memds)
    tempds = None
    memds = None


def IsTileNull(dataset, cropsize, xoff, yoff):
    """export one image batch sample"""
    driver = gdal.GetDriverByName("MEM")
    mem_ds = driver.Create("", cropsize, cropsize, 1, gdal.GDT_Byte)
    scanline = dataset.ReadRaster(xoff, yoff, cropsize, cropsize, cropsize, cropsize, gdal.GDT_Byte)
    mem_ds.WriteRaster(0, 0, cropsize, cropsize, scanline, cropsize, cropsize, gdal.GDT_Byte)
    array = mem_ds.GetRasterBand(1).ReadAsArray()
    max = np.max(array)
    if max <= 0:
        return 'Not'
    else:
        return 'Yes'


def ExportpngTile(dataset, adfGeoTransform, cropsize, xoff, yoff, pngpath):
    """export one image batch sample"""
    driver = gdal.GetDriverByName("MEM")
    mem_ds = driver.Create("", cropsize, cropsize, 1, gdal.GDT_Byte)
    scanline = dataset.ReadRaster(xoff, yoff, cropsize, cropsize, cropsize, cropsize, gdal.GDT_Byte)
    mem_ds.WriteRaster(0, 0, cropsize, cropsize, scanline, cropsize, cropsize, gdal.GDT_Byte)
    array = mem_ds.GetRasterBand(1).ReadAsArray()
    width, height = mem_ds.GetRasterBand(1).XSize, mem_ds.GetRasterBand(1).YSize
    mem_ds.SetGeoTransform(adfGeoTransform)
    mem_ds.SetProjection(dataset.GetProjection())
    drv = gdal.GetDriverByName("PNG")
    dst_ds = drv.CreateCopy(pngpath, mem_ds)
    dst_ds = None
    mem_ds = None
    return 'successful'


def ExportImageTile(dataset, adfgeotransform, cropsize, xoff, yoff, imagepath):
    """export one image batch sample"""
    driver = gdal.GetDriverByName("MEM")
    mem_ds = driver.Create("", cropsize, cropsize, 3, gdal.GDT_Byte)
    scanline = dataset.ReadRaster(xoff, yoff, cropsize, cropsize, cropsize, cropsize, gdal.GDT_Byte)
    mem_ds.WriteRaster(0, 0, cropsize, cropsize, scanline, cropsize, cropsize, gdal.GDT_Byte)
    array = mem_ds.GetRasterBand(1).ReadAsArray()
    width, height = mem_ds.GetRasterBand(1).XSize, mem_ds.GetRasterBand(1).YSize
    array[array > 5] = 1
    valpixelNum = float(array.sum())
    if valpixelNum / (width * height) < 0.2:
        return None
    mem_ds.SetGeoTransform(adfgeotransform)
    mem_ds.SetProjection(dataset.GetProjection())
    drv = gdal.GetDriverByName("JPEG")
    dst_ds = drv.CreateCopy(imagepath, mem_ds)
    dst_ds = None
    mem_ds = None
    return 'successful'


def ExportImageByPos(dataset, adfgeotransform, pos, cropsize, imagepath):
    driver = gdal.GetDriverByName("MEM")
    mem_ds = driver.Create("", cropsize, cropsize, 3, gdal.GDT_Byte)
    scanline = dataset.ReadRaster(pos[0], pos[1], pos[2], pos[3], cropsize, cropsize, gdal.GDT_Byte)
    mem_ds.WriteRaster(0, 0, cropsize, cropsize, scanline, cropsize, cropsize, gdal.GDT_Byte)
    array = mem_ds.GetRasterBand(1).ReadAsArray()
    width, height = mem_ds.GetRasterBand(1).XSize, mem_ds.GetRasterBand(1).YSize
    array[array > 5] = 1
    valpixelNum = float(array.sum())
    if valpixelNum / (width * height) < 0.2:
        return None
    mem_ds.SetGeoTransform(adfgeotransform)
    mem_ds.SetProjection(dataset.GetProjection())
    drv = gdal.GetDriverByName("JPEG")
    dst_ds = drv.CreateCopy(imagepath, mem_ds)
    dst_ds = None
    mem_ds = None
    return 'successful'


def GetPoseList(before_ds, after_ds, cropsize, stridesize=256):
    preTransform = before_ds.GetGeoTransform()
    proTransform = after_ds.GetGeoTransform()
    prestartx, prestarty = preTransform[0], preTransform[3]
    prewidth, preheight = before_ds.GetRasterBand(1).XSize, before_ds.GetRasterBand(1).YSize
    preendx, preendy = ImageRowCol2Projection(preTransform, prewidth - 1, preheight - 1)

    prostartx, prostarty = proTransform[0], proTransform[0]
    prowidth, proheight = after_ds.GetRasterBand(1).XSize, after_ds.GetRasterBand(1).YSize
    proendx, proendy = ImageRowCol2Projection(proTransform, prowidth - 1, proheight - 1)
    interStartX = max(prestartx, prostartx)
    interStartY = min(prestarty, prostarty)
    interEndX = min(preendx, proendx)
    interEndY = max(preendy, proendy)

    xresolution = (preTransform[1] + proTransform[1]) / 2.0
    yresolution = (preTransform[5] + proTransform[5]) / 2.0
    newTransform = (interStartX, xresolution, 0, interStartY, 0, yresolution)
    width, height = Projection2ImageRowCol(newTransform, interEndX, interEndY)
    width += 1
    height += 1
    # stridesize = int(256)
    rows_max = int((height - cropsize) / stridesize) + 1
    columns_max = int((width - cropsize) / stridesize) + 1

    preList = []
    proList = []
    targetList = []
    for i in range(rows_max + 1):
        for j in range(columns_max + 1):
            x_off = width - cropsize if j == columns_max else j * stridesize
            y_off = height - cropsize if i == rows_max else i * stridesize

            batchstartx, batchstarty = ImageRowCol2Projection(newTransform, x_off, y_off)
            batchendx, batchendy = ImageRowCol2Projection(newTransform, x_off + cropsize - 1, y_off + cropsize - 1)
            targetList.append([batchstartx, batchstarty, batchendx, batchendy])
            # pre
            pre_xoff, pre_yoff = Projection2ImageRowCol(preTransform, batchstartx, batchstarty)
            pre_cropx, pre_cropy = Projection2ImageRowCol(preTransform, batchendx, batchendy)
            pre_cropx, pre_cropy = pre_cropx - pre_xoff + 1, pre_cropy - pre_yoff + 1
            preList.append([pre_xoff, pre_yoff, pre_cropx, pre_cropy])
            # pro
            pro_xoff, pro_yoff = Projection2ImageRowCol(proTransform, batchstartx, batchstarty)
            pro_cropx, pro_cropy = Projection2ImageRowCol(proTransform, batchendx, batchendy)
            pro_cropx, pro_cropy = pro_cropx - pro_xoff + 1, pro_cropy - pro_yoff + 1
            proList.append([pro_xoff, pro_yoff, pro_cropx, pro_cropy])
    return preList, proList, targetList


def savegrid(targetlist, srs, outshp):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.CreateDataSource(outshp)
    layer = ds.CreateLayer('Polygon', srs, ogr.wkbPolygon)
    for tl in targetlist:
        startX, startY, endX, endY = tl[0], tl[1], tl[2], tl[3]
        wkt = "POLYGON((%s %s,%s %s,%s %s,%s %s,%s %s))" % (
            startX, startY, endX, startY, endX, endY, startX, endY, startX, startY)
        geometry = ogr.CreateGeometryFromWkt(wkt)
        desfDef = layer.GetLayerDefn()
        desF = ogr.Feature(desfDef)
        desF.SetGeometry(geometry)
        layer.CreateFeature(desF)
    ds = None


def Projection2ImageRowCol(adfGeoTransform, dProjX, dProjY):
    """translate from projection to image row and col"""
    dTemp = adfGeoTransform[1] * adfGeoTransform[5] - adfGeoTransform[2] * adfGeoTransform[4]
    dCol = (adfGeoTransform[5] * (dProjX - adfGeoTransform[0]) - adfGeoTransform[2] * (
            dProjY - adfGeoTransform[3])) / dTemp + 0.5
    dRow = (adfGeoTransform[1] * (dProjY - adfGeoTransform[3]) - adfGeoTransform[4] * (
            dProjY - adfGeoTransform[0])) / dTemp + 0.5
    return int(dCol), int(dRow)


def ImageRowCol2Projection(adfGeoTransform, iCol, iRow):
    """translate from image row and col to projection"""
    dProjX = adfGeoTransform[0] + adfGeoTransform[1] * iCol + adfGeoTransform[2] * iRow
    dProjY = adfGeoTransform[3] + adfGeoTransform[4] * iCol + adfGeoTransform[5] * iRow
    return dProjX, dProjY
