'''
TODO:
    * Exploratory
        * Implement data import from H5
            - Drop-down menu: Import dataset
            - Drop-down menu: Append as column
    * Data
        * Save state to H5
            - store dataframes, data, and data axis labels
        * Read state from H5
        * Implement multiple datasets into gui. Switch between datasets
        * Implement queries
    * Plotting
        * Only plot selected datasets
        * Add channel and trial labels to colors
'''


###############################
# Import system libraries
###############################
import os, sys, locale
import json
import numpy as np
import pandas as pd
from PyQt5 import QtGui, QtCore, QtWidgets

#######################################
# Import local libraries
#######################################

from mesostat.utils.qt_gui_helper import compile_form, qtable_load_from_pandas
from mesostat.utils.system import getfiles, getfiles_walk
from mesostat.utils.strings import path_extract_data
from mesostat.utils.dictionaries import merge_dicts_of_lists
from mesostat.utils.matlab_helper import loadmat
from mesostat.visualization.mpl_qt import MplPlotLayout
from mesostat.visualization.embeddings import plot_pca
from mesostat.gui.h5_explorer import H5Explorer

#######################################
# Compile QT File
#######################################
thisdir = os.path.dirname(os.path.abspath(__file__))
compile_form(os.path.join(thisdir, 'dataload.ui'))
compile_form(os.path.join(thisdir, 'importh5.ui'))
compile_form(os.path.join(thisdir, 'mesostatgui.ui'))

#######################################
# Load compiled forms
#######################################
from mesostat.gui.dataload import Ui_DataLoad
from mesostat.gui.importh5 import Ui_importH5Dialog
from mesostat.gui.mesostatgui import Ui_MesostatGui


#######################################################
# Main Window
#######################################################
class MesostatGUI():
    def __init__(self, dialog):
        # Init
        self.dialog = dialog
        self.gui = Ui_MesostatGui()
        self.gui.setupUi(dialog)

        # GUI-Constants
        self.fontsize = 15
        self.pathSettings = os.path.join(thisdir, 'settings.json')

        # Variables
        self.data = []
        self.df = pd.DataFrame()

        # Objects and initialization
        self._init_data_queries()
        self._init_plot_layout()
        self._react_plot_timemode_change()  # Init sliders in the correct state

        # Reacts
        self.gui.plotParamModeComboBox.currentIndexChanged.connect(self._react_plot_timemode_change)
        self.gui.plotButton.clicked.connect(self.plot_data)
        self.gui.explorerTableWidget.doubleClicked.connect(self.exlorer_move_h5)
        self.gui.explorerTableWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.gui.explorerTableWidget.customContextMenuRequested.connect(self._react_explorer_dropdown_menu)

        # File menu actions
        self.gui.actionLoad_data.triggered.connect(self.load_data_gui)
        self.gui.actionLoad_H5.triggered.connect(self.explorer_load_h5)
        self.gui.actionLoad_data.setShortcut('Ctrl+o')
        self.gui.actionLoad_H5.setShortcut('Ctrl+h')

    def json_load(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def json_write(self, path, dict):
        with open(path, 'w') as f:
            json.dump(dict, f)

    #############################
    #  Explorer
    #############################

    def _react_explorer_dropdown_menu(self, point):
        self._explMenu = QtWidgets.QMenu(self.gui.explorerTableWidget)
        actionAddAsDfCol = self._explMenu.addAction('Add as df column')
        actionAddAsData = self._explMenu.addAction('Add as data')
        actionAddAsAxisLabel = self._explMenu.addAction('Add as axis label')
        actionAddAsDfCol.triggered.connect(self.explorer_add_data_as_df_col)
        actionAddAsData.triggered.connect(self.explorer_add_data_as_data)
        actionAddAsAxisLabel.triggered.connect(self.explorer_add_data_as_axis_label)
        self._explMenu.popup(self.gui.explorerTableWidget.viewport().mapToGlobal(point))

    def explorer_load_h5(self):
        settings = self.json_load(self.pathSettings)
        fname = QtWidgets.QFileDialog.getOpenFileName(self.dialog, "Load H5 file", settings['path_data'],
                                                      'HDF5 (*.h5)', options=QtWidgets.QFileDialog.DontUseNativeDialog)[0]

        self.h5exp = H5Explorer(fname)
        self.explorer_update_table()

        settings['path_data'] = fname
        self.json_write(self.pathSettings, settings)

    def explorer_update_table(self):
        innerPath = self.h5exp.get_inner_path()
        self.gui.explorerInnerPathLineEdit.setText(innerPath)

        dfThis = pd.DataFrame(self.h5exp.contents_current())
        if innerPath != '/':
            dfThis = pd.concat([pd.DataFrame({'label': '..', 'type': 'Group', 'shape' : ''}, index=[0]), dfThis], ignore_index=True)

        qtable_load_from_pandas(self.gui.explorerTableWidget, dfThis)

    def exlorer_move_h5(self):
        selectedItems = self.gui.explorerTableWidget.selectedItems()
        assert len(selectedItems) == 3
        label = selectedItems[0].text()
        elemType = selectedItems[1].text()

        if elemType == 'Group':
            if label == '..':
                errcode, errMsg = self.h5exp.move_parent()
            else:
                errcode, errMsg = self.h5exp.move_child(label)

            print(errcode, errMsg)
            self.explorer_update_table()

    def explorer_add_data_as_df_col(self):
        pass

    def explorer_add_data_as_data(self):
        self.importH5Dialog = QtWidgets.QWidget()
        self.importH5Dialog.show()
        self.importH5Gui = Ui_importH5Dialog()
        self.importH5Gui.setupUi(self.importH5Dialog)

        # # Actions:
        # self.importH5Gui.newDatasetRadioButton.changeEvent.connect(self.load_data_impl)
        # self.importH5Gui.dataLoadCancelButton.clicked.connect(lambda : self.loadDataDialog.close())
        # self.importH5Gui.pathDataButton.clicked.connect(self.get_path_data)

    def explorer_add_data_as_axis_label(self):
        pass

    #############################
    #  Data
    #############################

    def _init_data_queries(self):
        self.dataQueryButtons = []
        self.dataQueryLayout = QtWidgets.QGridLayout()
        self.gui.dataQueryGroupBox.setLayout(self.dataQueryLayout)

        tmpButton = QtWidgets.QPushButton('+', self.dialog)
        self.dataQueryButtons += [tmpButton]

        self.dataQueryLayout.addWidget(tmpButton, 0, 0)

    def load_data_gui(self):
        # Init
        self.loadDataDialog = QtWidgets.QWidget()
        self.loadDataDialog.show()
        self.loadDataGUI = Ui_DataLoad()
        self.loadDataGUI.setupUi(self.loadDataDialog)

        # Actions:
        self.loadDataGUI.dataLoadOkButton.clicked.connect(self.load_data_impl)
        self.loadDataGUI.dataLoadCancelButton.clicked.connect(lambda : self.loadDataDialog.close())
        self.loadDataGUI.pathDataButton.clicked.connect(self.get_path_data)

        # self.newDeckGUI.newDeckDuplicatePushButton.clicked.connect(self.import_deck_gui)
        # self.newDeckGUI.newDeckCancelPushButton.clicked.connect(self.newDeckDialog.close)
        # self.newDeckGUI.newDeckOkPushButton.clicked.connect(self.new_deck)

    def load_data_impl(self):
        ################################
        # Extract info from form
        ################################
        dataPath = self.loadDataGUI.pathDataLineEdit.text()
        haveDeepSearch = self.loadDataGUI.pathRecursiveCheckBox.isChecked()
        fnameTemplate = self.loadDataGUI.parseFnameLineEdit.text()
        dataInsideKey = self.loadDataGUI.fileStructureLineEdit.text()
        ftype = '.mat' if self.loadDataGUI.fileTypeComboBox.currentIndex() == 0 else '.h5'
        extraKeys = [k.strip() for k in self.loadDataGUI.pathFilterKeysLineEdit.text().split(',')]
        allKeys = [ftype] + extraKeys

        ################################
        # Get data file names
        ################################
        if haveDeepSearch:
            fileswalk = getfiles_walk(dataPath, allKeys)
            filepaths = [os.path.join(dir, file) for dir, file in fileswalk]
        else:
            files = getfiles(dataPath, allKeys)
            filepaths = [os.path.join(dataPath, file) for file in files]

        ################################
        # Parse data file names
        ################################
        if fnameTemplate == '':
            df = pd.DataFrame({'fpaths' : filepaths})
        else:
            df = pd.DataFrame(merge_dicts_of_lists([path_extract_data(s, fnameTemplate) for s in filepaths]))

        ################################
        # Read data
        ################################
        self.data = []
        self.loadDataGUI.loadDataProgressBar.setEnabled(True)
        self.loadDataGUI.loadDataProgressBar.setValue(0)
        self.loadDataGUI.loadDataProgressBar.setMaximum(len(filepaths))
        if ftype == '.mat':
            for iPath, fpath in enumerate(filepaths):
                self.data += [loadmat(fpath)[dataInsideKey][:, :, :48]]
                self.loadDataGUI.loadDataProgressBar.setValue(iPath+1)
        else:
            raise ValueError('Not Implemented')

        ################################
        # Update data table
        ################################
        df['dataname'] = 'src'
        df['shape'] = [str(d.shape) for d in self.data]
        self.df = df
        qtable_load_from_pandas(self.gui.dataTableWidget, df)

        ################################
        # Update plot coloring
        ################################
        self.dataKeys = set(df.columns) - {'dataname', 'shape'}
        self.gui.plotColorComboBox.clear()
        self.gui.plotColorComboBox.addItems(['None'] + list(self.dataKeys))

        self.loadDataDialog.close()

    def get_path_data(self):
        settings = self.json_load(self.pathSettings)

        fpath = QtWidgets.QFileDialog.getExistingDirectory(self.loadDataDialog, "Get data directory", settings['path_data'])
        if fpath != "":
            self.loadDataGUI.pathDataLineEdit.setText(fpath)

            settings['path_data'] = fpath
            self.json_write(self.pathSettings, settings)

    #############################
    #  Plots
    #############################

    def _init_plot_layout(self):
        self.plotlayout = MplPlotLayout(self.dialog, width=5, height=4, dpi=100)
        layout = self.gui.plotsTab.layout()

        layout.addWidget(self.plotlayout.widget)
        self.plotlayout.canvas.axes.plot([0,1,2,3,4], [10,1,20,3,40])

    def _react_plot_timemode_change(self):
        isExact = self.gui.plotParamModeComboBox.currentText() == 'exact'
        self.gui.plotMinParamSlider.setEnabled(not isExact)
        self.gui.plotMaxParamSlider.setEnabled(not isExact)
        self.gui.plotExactParamSlider.setEnabled(isExact)

    def plot_data(self):
        dataOrd = self.gui.plotDataOrderComboBox.currentText()
        redParam = self.gui.plotReducedParamComboBox.currentText()
        idxReduced = dataOrd.index(redParam)
        colorKey = self.gui.plotColorComboBox.currentText()

        dataEff = []
        colorValLst = []
        for idx, row in self.df.iterrows():
            dataEff += [np.mean(self.data[idx], axis=idxReduced)]

            if colorKey != 'None':
                colorValLst += [row[colorKey]]*len(dataEff[-1])
        dataEff = np.vstack(dataEff)

        labels = np.array(colorValLst) if colorKey != 'None' else None

        self.plotlayout.canvas.axes.clear()
        plot_pca(self.plotlayout.canvas.axes, dataEff, labels=labels)


#######################################################
## Start the QT window
#######################################################
if __name__ == '__main__' :
    app = QtWidgets.QApplication(sys.argv)
    mainwindow = QtWidgets.QMainWindow()
    locale.setlocale(locale.LC_TIME, "en_GB.utf8")
    classInstance = MesostatGUI(mainwindow)
    mainwindow.show()
    sys.exit(app.exec_())
