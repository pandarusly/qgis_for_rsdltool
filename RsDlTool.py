# -*- coding: utf-8 -*-
"""
/***************************************************************************
 RSDlTool
                                 A QGIS plugin
 This is Plugin use DL in Rs
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2022-08-31
        git sha              : $Format:%H$
        copyright            : (C) 2022 by libin
        email                : ts20160039a31@edu.cumt.cn
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
import os.path
import tempfile
from typing import Union, List

from PyQt5.QtCore import QThread
# Initialize Qt resources from file resources.py
from PyQt5.QtWidgets import (
    QMessageBox, QFileDialog, QAction
)
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsRasterLayer, Qgis, QgsMessageLog, QgsGeometry, QgsCoordinateReferenceSystem,
    QgsFeature, QgsMapLayer, QgsDistanceArea
)
from qgis.gui import QgsFileWidget

from . import helpers, config
from .RsDlTool_dialog import RSDlToolDialog
from .workers import SplittingCreator, DetectionCreator
from .resources import *


class RSDlTool:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        self.plugin_name = 'RSDLTOOL'

        self.temp_dir = tempfile.gettempdir()
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'RSDlTool_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.toolbar = self.iface.addToolBar(config.PLUGIN_NAME)
        self.toolbar.setObjectName(config.PLUGIN_NAME)
        self.menu = self.tr(u'&Rs DL Tool')
        # Check if plugin was started the first time in current QGIS session
        # Must be set in initGui() to survive plugin reloads
        self.first_start = None

    def valid_exist_layer(self, path):
        # 判断是否有同样名字
        rlayer = self.project.mapLayersByName(os.path.splitext(os.path.basename(path))[0])
        if len(rlayer) > 0:
            return True
        else:
            return False 

    def select_tif1_dataset(self) -> None:
        dlg = QFileDialog(self.dlg, self.tr("Select Image"))
        dlg.setMimeTypeFilters(['image/tiff', "application/octet-stream", 'image/png', 'image/jpeg'])
        if dlg.exec():
            path: str = dlg.selectedFiles()[0]
            exit_layer = self.valid_exist_layer(path)
            if not exit_layer:
                layer = QgsRasterLayer(path, os.path.splitext(os.path.basename(path))[0])
            else:
                layer = QgsRasterLayer(path,
                                       os.path.splitext(os.path.basename(path))[0] + "_{}".format(os.urandom(2).hex()))
            self.project.addMapLayer(layer)
            self.dlg.rDSACombo.setLayer(layer)

    def select_tif2_dataset(self) -> None:
        dlg = QFileDialog(self.dlg, self.tr("Select Post Image"))
        dlg.setMimeTypeFilters(['image/tiff', "application/octet-stream", 'image/png', 'image/jpeg'])
        if dlg.exec():
            path: str = dlg.selectedFiles()[0]
            layer = self.valid_exist_layer(path)
            if not layer:
                layer = QgsRasterLayer(path, os.path.splitext(os.path.basename(path))[0])
            else:
                layer = QgsRasterLayer(path,
                                       os.path.splitext(os.path.basename(path))[0] + "_{}".format(os.urandom(2).hex()))
            self.project.addMapLayer(layer)
            self.dlg.rDSBCombo.setLayer(layer)

    def select_vector_dataset(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self.dlg, "Select Label File", "", "*.shp;*.png;*.tif"
        )
        layer = self.valid_exist_layer(path)
        if os.path.exists(path):
            if not layer:
                if ".shp" in path:
                    layer = QgsVectorLayer(path, os.path.splitext(os.path.basename(path))[0])
                else:
                    layer = QgsRasterLayer(path, os.path.splitext(os.path.basename(path))[0])
            else:
                if ".shp" in path:
                    layer = QgsVectorLayer(path, os.path.splitext(os.path.basename(path))[0] + "_{}".format(
                        os.urandom(2).hex()))
                else:
                    layer = QgsRasterLayer(path, os.path.splitext(os.path.basename(path))[0] + "_{}".format(
                        os.urandom(2).hex()))
            self.project.addMapLayer(layer)
            self.dlg.vDSCombo.setLayer(layer)

    def select_mask_dataset(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self.dlg, "Select ESRI Shpfiles", "", "*.shp"
        )
        layer = self.valid_exist_layer(path)
        if not layer:
            layer = QgsVectorLayer(path, os.path.splitext(os.path.basename(path))[0])
        else:
            layer = QgsVectorLayer(path,
                                   os.path.splitext(os.path.basename(path))[0] + "_{}".format(os.urandom(2).hex()))
        self.project.addMapLayer(layer)
        self.dlg.vDSMaskCombo.setLayer(layer)

    def calculate_aoi_area_polygon_layer(self, layer: Union[QgsVectorLayer, None]) -> None:
        """Get the AOI size total when polygon another layer is chosen,
        current layer's selection is changed or the layer's features are modified.

        :param layer: The current polygon layer
        """
        if not self.dlg.checkBoxMask.isChecked():  # GeoTIFF extent used; no difference
            self.calculate_aoi_area_raster(self.dlg.rDSACombo.currentLayer())
            return
        if layer and layer.featureCount() > 0:
            features = list(layer.getSelectedFeatures()) or list(layer.getFeatures())
            if len(features) == 1:
                aoi = features[0].geometry()
            else:
                aoi = QgsGeometry.collectGeometry([feature.geometry() for feature in features])
            self.calculate_aoi_area(aoi, layer.crs())
        else:  # empty layer or combo's itself is empty
            self.dlg.labelAoiArea.clear()
            self.aoi = self.aoi_size = None

    def calculate_aoi_area_raster(self, layer: Union[QgsRasterLayer, None]) -> None:
        """Get the AOI size when a new entry in the raster combo box is selected.

        :param layer: The current raster layer
        """
        if layer:
            geometry = QgsGeometry.collectGeometry([QgsGeometry.fromRect(layer.extent())])
            self.calculate_aoi_area(geometry, layer.crs())
        else:
            self.calculate_aoi_area_polygon_layer(self.dlg.vDSMaskCombo.currentLayer())

    def calculate_aoi_area_use_image_extent(self, use_image_extent: bool) -> None:
        """Get the AOI size when the Use image extent checkbox is toggled.

        :param use_image_extent: The current state of the checkbox
        """
        if not use_image_extent:
            self.calculate_aoi_area_raster(self.dlg.rDSACombo.currentLayer())
        else:
            self.calculate_aoi_area_polygon_layer(self.dlg.vDSMaskCombo.currentLayer())

    def calculate_aoi_area_selection(self, _: List[QgsFeature]) -> None:
        """Get the AOI size when the selection changed on a polygon layer.

        :param _: A list of currently selected features
        """
        layer = self.dlg.vDSMaskCombo.currentLayer()
        if layer == self.iface.activeLayer():
            self.calculate_aoi_area_polygon_layer(layer)

    def calculate_aoi_area_layer_edited(self) -> None:
        # 暂时不用
        """Get the AOI size when a feature is added or remove from a layer."""
        layer = self.sender()  # bug
        if layer == self.dlg.vDSMaskCombo.currentLayer():
            self.calculate_aoi_area_polygon_layer(layer)

    def calculate_aoi_area(self, aoi: QgsGeometry, crs: QgsCoordinateReferenceSystem) -> None:
        """Display the AOI size in sq.km.

        :param aoi: the processing area.
        :param crs: the CRS of the processing area.
        """
        if crs != helpers.WGS84:
            aoi = helpers.to_wgs84(aoi, crs)
        self.aoi = aoi  # save for reuse in processing creation or metadata requests
        # Set ellipsoid to calculate on sphere if the CRS is geographic
        self.calculator.setEllipsoid(helpers.WGS84_ELLIPSOID)
        self.calculator.setSourceCrs(helpers.WGS84, self.project.transformContext())
        self.aoi_size = self.calculator.measureArea(aoi) / 10 ** 6  # sq. m to sq.km
        self.dlg.labelAoiArea.setText(self.tr('Area: {:.2f} sq.km').format(self.aoi_size))

    def monitor_polygon_layer_feature_selection(self, layers: List[QgsMapLayer]) -> None:
        """Set up connection between feature selection in polygon layers and AOI area calculation.

        Since the plugin allows using a single feature withing a polygon layer as an AOI for processing,
        its area should then also be calculated and displayed in the UI, just as with a single-featured layer.
        For every polygon layer added to the project, this function sets up a signal-slot connection for
        monitoring its feature selection by passing the changes to calculate_aoi_area().

        :param layers: A list of layers of any type (all non-polygon layers will be skipped)
        """
        for layer in filter(helpers.is_polygon_layer, layers):
            layer.selectionChanged.connect(self.calculate_aoi_area_selection)
            # layer.editingStopped.connect(self.calculate_aoi_area_layer_edited)

    def alert(self, message: str, kind: str = 'information') -> None:
        """Display an interactive modal pop up.

        :param message: A text to display
        :param kind: The type of a pop-up to display; it is translated into a class method name of QMessageBox,
            so must be one of https://doc.qt.io/qt-5/qmessagebox.html#static-public-members
        """
        return getattr(QMessageBox, kind)(self.dlg, self.plugin_name, message)

    def push_message(self, message: str, level: Qgis.MessageLevel = Qgis.Info, duration: int = 5) -> None:
        """Display a message on the message bar.

        :param message: A text to display
        :param level: The type of a message to display
        :param duration: For how long the message will be displayed
        """
        self.iface.messageBar().pushMessage(config.PLUGIN_NAME, message, level, duration)

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('RSDlTool', message)

    def add_action(
            self,
            icon_path,
            text,
            callback,
            enabled_flag=True,
            add_to_menu=True,
            add_to_toolbar=True,
            status_tip=None,
            whats_this=None,
            parent=None): 
        
        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            # Adds plugin icon to Plugins toolbar
            self.iface.addToolBarIcon(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/RsDlTool/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u''),
            callback=self.run,
            parent=self.iface.mainWindow())

        # will be set False in run()
        self.first_start = True

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&Rs DL Tool'),
                action)
            self.iface.removeToolBarIcon(action)

    def log(self, message: str, level: Qgis.MessageLevel = Qgis.Warning) -> None:
        """Log a message to the QGIS Message Log.

        :param message: A text to display
        :param level: The type of a message to display
        """
        QgsMessageLog.logMessage(message, config.PLUGIN_NAME, level=level)

    def get_layer_path(self, mapLayerCombox):
        currentlayer = mapLayerCombox.currentText().split(" ")[0]
        if len(currentlayer) < 1:
            return ""
        rlayer = self.project.mapLayersByName(currentlayer)
        path = rlayer[0].dataProvider().dataSourceUri()
        return path

    def log_finished(self):
        self.alert(self.tr('Finished!'))

    def show_result(self, show_list):
        for show_dict in show_list:
            for k, v in show_dict.items():
                if os.path.exists(v):
                    if k == "binary" and self.dlg.checkBoxSB.isChecked():
                        self.iface.addRasterLayer(
                            v, os.path.splitext(os.path.basename(v))[0])
                    if k == "muticlass" and self.dlg.checkBoxMutiCls.isChecked():
                        self.iface.addRasterLayer(v, os.path.splitext(os.path.basename(v))[0])
                    if k == "shp_path" and self.dlg.checkShowShpBox.isChecked():
                        vec_layer = QgsVectorLayer(v, os.path.splitext(os.path.basename(v))[0])
                        self.project.addMapLayer(vec_layer)
                    if k == "png_path" and self.dlg.checkShowPngBox.isChecked():
                        ras_layer = QgsRasterLayer(v, os.path.splitext(os.path.basename(v))[0])
                        self.project.addMapLayer(ras_layer)

    def create_splitting(self):
        a_series_path = self.dlg.mQgsFileWidget_RA.filePath()
        a_series_path = sorted(QgsFileWidget.splitFilePaths(a_series_path))
        b_series_path = self.dlg.mQgsFileWidget_RB.filePath()
        b_series_path = sorted(QgsFileWidget.splitFilePaths(b_series_path))
        v_series_path = self.dlg.mQgsFileWidget_V.filePath()
        v_series_path = sorted(QgsFileWidget.splitFilePaths(v_series_path))
        a_ras_path = self.get_layer_path(self.dlg.rDSACombo)
        b_ras_path = self.get_layer_path(self.dlg.rDSBCombo) if self.dlg.useChangeCheck.isChecked() else None
        vec_path = self.get_layer_path(self.dlg.vDSCombo) if self.dlg.checkBoxSB_2.isChecked() or self.dlg.checkBoxMutiCls_2.isChecked() else None
        mask_path = self.get_layer_path(self.dlg.vDSMaskCombo) if self.dlg.checkBoxMask.isChecked() else None
        classCfgPath = self.dlg.mQfwDataset_ClassCfg.filePath() if self.dlg.checkBoxMutiCls_2.isChecked() else None
        datasetOutDir = self.dlg.mQgsFileWidget_DatasetOut.filePath()
        SplittingBlockSize = int(self.dlg.blockCombo.currentText())
        overlap = self.dlg.StrideCombo.opacity()
        SplittingStrideSize = int(SplittingBlockSize * (1 - overlap))
        uniform_name = self.dlg.lineEditUN.text()
        ui_params = {
            'a_series_path': a_series_path,
            'b_series_path': b_series_path,
            'v_series_path': v_series_path,
            'a_ras_path': a_ras_path,
            'b_ras_path': b_ras_path,
            'vec_path': vec_path,
            'mask_path': mask_path,
            'classCfgPath': classCfgPath,
            'datasetOutDir': datasetOutDir,
            'SplittingBlockSize': SplittingBlockSize,
            'SplittingStrideSize': SplittingStrideSize,
            'uniform_name': uniform_name,
            'useBatchCheck': self.dlg.useBatchCheck.isChecked(),
            'checkBoxSB_2': self.dlg.checkBoxSB_2.isChecked(),
            'useChangeCheck': self.dlg.useChangeCheck.isChecked(),
            'temp_dir': os.path.join(self.temp_dir, "processing_tfYpER", os.urandom(32).hex()),
            'color_text_path': os.path.join(self.plugin_dir + "/utils/color.txt"),
            'youhua': dict(valpixel=float(self.dlg.lineEdit_valpixel.text()),
                           threshold=int(self.dlg.lineEdit_threshold.text()),
                           len_contours=int(self.dlg.lineEdit_len_contours.text())
                           )
        }
        # Spin up a worker, a thread, and move the worker to the thread
        thread = QThread(self.iface.mainWindow())
        worker = SplittingCreator(params=ui_params)
        worker.moveToThread(thread)
        thread.started.connect(worker.create_splitting)
        worker.finished.connect(thread.quit)
        worker.finished.connect(self.log_finished)
        worker.fetched.connect(lambda show_list: self.show_result(show_list))
        worker.error.connect(lambda error: self.log(error))
        worker.error.connect(
            lambda: self.alert(self.tr('Processing creation failed, see the QGIS log for details'), kind='critical'))
        self.dlg.finished.connect(thread.requestInterruption)
        thread.start()
        self.push_message(self.tr('Starting the processing...'))

    def dataset_maker(self):
        self.dlg.selectVDS.clicked.connect(self.select_vector_dataset)
        class_cfg_path = os.path.join(self.plugin_dir + "/utils/tblx.yaml")
        self.dlg.mQfwDataset_ClassCfg.lineEdit().setText(class_cfg_path)
        # -----开始运行
        self.dlg.startSplitting.clicked.connect(self.create_splitting)

    def create_detection(self):
        a_series_path = self.dlg.mQgsFileWidget_RA.filePath()
        a_series_path = sorted(QgsFileWidget.splitFilePaths(a_series_path))
        b_series_path = self.dlg.mQgsFileWidget_RB.filePath()
        b_series_path = sorted(QgsFileWidget.splitFilePaths(b_series_path))

        ui_params = dict(
            a_series_path=a_series_path,
            b_series_path=b_series_path,
            use_exe=self.dlg.mQgsFileWidget_USEEXE.filePath(),
            useBatchCheck=self.dlg.useBatchCheck.isChecked(),
            useChangeDect=self.dlg.useChangeCheck.isChecked(),
            CfgPathName=self.dlg.mQgsFileWidget_CfgPath.filePath(),
            processing_name=self.dlg.processingName.text() + '_' + str(os.urandom(4).hex()),
            modelPathName=self.dlg.mQgsFileWidget_ModelPath.filePath(),
            rasterComboA=self.get_layer_path(self.dlg.rDSACombo),
            rasterComboB=self.get_layer_path(self.dlg.rDSBCombo),
            targetDirName=self.dlg.mQgsFileWidget_TarDir.filePath(),
            checkSHPBox=self.dlg.checkSHPBox.isChecked(),
            checkPNGBox=self.dlg.checkPNGBox.isChecked(),
            checkHALFBox=self.dlg.checkHALFBox.isChecked(),
            STRIDE_SIZE=int(self.dlg.STRIDELineEdit.text()),
            BLOCKSIZE=int(self.dlg.BLOCKLineEdit.text()),
            BATCHSIZE=int(self.dlg.BatchSizeLineEdit.text()),
            USECUDA=self.dlg.checkCUDABox.isChecked(),
            WEIGHT=self.dlg.weightCombo.currentText(),
            maxval=int(self.dlg.MaxvalLineEdit.text()),
            simplify_ratiao=float(self.dlg.SimplifyLineEdit.text()),
            mask_path=self.get_layer_path(self.dlg.vDSMaskCombo) if self.dlg.checkBoxMask.isChecked() else None
        )
        # Spin up a worker, a thread, and move the worker to the thread
        thread = QThread(self.iface.mainWindow())
        worker = DetectionCreator(params=ui_params)
        worker.moveToThread(thread)
        thread.started.connect(worker.create_detection)
        worker.finished.connect(thread.quit)
        worker.finished.connect(self.log_finished)
        worker.fetched.connect(lambda show_list: self.show_result(show_list))
        worker.error.connect(lambda error: self.log(error))
        worker.error.connect(
            lambda: self.alert(self.tr('Processing creation failed, see the QGIS log for details'), kind='critical'))
        self.dlg.finished.connect(thread.requestInterruption)
        thread.start()
        self.push_message(self.tr('Starting the processing...'))

    def inference(self):
        self.dlg.weightCombo.setCurrentIndex(0)
        self.dlg.startDetcProcessing.clicked.connect(self.create_detection)

    def run(self):
        """Run method that performs all the real work"""

        # Create the dialog with elements (after translation) and keep reference
        # Only create GUI ONCE in callback, so that it will only load when the plugin is started
        if self.first_start == True:
            self.first_start = False

            self.project = QgsProject.instance()  # 用self.project 代替 QgsProject..instance()
            self.dlg = RSDlToolDialog()
            # Calculate AOI size
            self.calculator = QgsDistanceArea()
            self.dlg.vDSMaskCombo.layerChanged.connect(self.calculate_aoi_area_polygon_layer)
            self.dlg.rDSACombo.layerChanged.connect(self.calculate_aoi_area_raster)
            self.dlg.checkBoxMask.toggled.connect(self.calculate_aoi_area_use_image_extent)
            # self.dlg.checkBoxMask.toggled.connect(self.calculate_aoi_area_use_image_extent)
            self.monitor_polygon_layer_feature_selection([
                self.project.mapLayer(layer_id) for layer_id in self.project.mapLayers(validOnly=True)
            ])
            self.project.layersAdded.connect(self.monitor_polygon_layer_feature_selection)
            # Connect buttons
            self.dlg.selectRDSA.clicked.connect(self.select_tif1_dataset)
            self.dlg.selectRDSB.clicked.connect(self.select_tif2_dataset)
            self.dlg.selectVDSMask.clicked.connect(self.select_mask_dataset)
            self.dataset_maker()
            self.inference()

        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            # Do something useful here - delete the line containing pass and
            # substitute with your code.
            pass
