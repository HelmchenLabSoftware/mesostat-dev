# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/alyosha/work/git/mesostat-dev/mesostat/gui/dataload.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_DataLoad(object):
    def setupUi(self, DataLoad):
        DataLoad.setObjectName("DataLoad")
        DataLoad.resize(492, 294)
        self.verticalLayout = QtWidgets.QVBoxLayout(DataLoad)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_3 = QtWidgets.QLabel(DataLoad)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 5, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(DataLoad)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 6, 0, 1, 1)
        self.parseFnameLineEdit = QtWidgets.QLineEdit(DataLoad)
        self.parseFnameLineEdit.setObjectName("parseFnameLineEdit")
        self.gridLayout.addWidget(self.parseFnameLineEdit, 5, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(DataLoad)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 3, 0, 1, 1)
        self.fileTypeComboBox = QtWidgets.QComboBox(DataLoad)
        self.fileTypeComboBox.setObjectName("fileTypeComboBox")
        self.fileTypeComboBox.addItem("")
        self.fileTypeComboBox.addItem("")
        self.gridLayout.addWidget(self.fileTypeComboBox, 0, 1, 1, 1)
        self.label = QtWidgets.QLabel(DataLoad)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.pathRecursiveCheckBox = QtWidgets.QCheckBox(DataLoad)
        self.pathRecursiveCheckBox.setObjectName("pathRecursiveCheckBox")
        self.gridLayout.addWidget(self.pathRecursiveCheckBox, 4, 1, 1, 1)
        self.pathDataButton = QtWidgets.QPushButton(DataLoad)
        self.pathDataButton.setObjectName("pathDataButton")
        self.gridLayout.addWidget(self.pathDataButton, 2, 0, 1, 1)
        self.fileStructureLineEdit = QtWidgets.QLineEdit(DataLoad)
        self.fileStructureLineEdit.setObjectName("fileStructureLineEdit")
        self.gridLayout.addWidget(self.fileStructureLineEdit, 6, 1, 1, 1)
        self.pathFilterKeysLineEdit = QtWidgets.QLineEdit(DataLoad)
        self.pathFilterKeysLineEdit.setObjectName("pathFilterKeysLineEdit")
        self.gridLayout.addWidget(self.pathFilterKeysLineEdit, 3, 1, 1, 1)
        self.pathDataLineEdit = QtWidgets.QLineEdit(DataLoad)
        self.pathDataLineEdit.setObjectName("pathDataLineEdit")
        self.gridLayout.addWidget(self.pathDataLineEdit, 2, 1, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.dataLoadCancelButton = QtWidgets.QPushButton(DataLoad)
        self.dataLoadCancelButton.setObjectName("dataLoadCancelButton")
        self.horizontalLayout.addWidget(self.dataLoadCancelButton)
        self.dataLoadOkButton = QtWidgets.QPushButton(DataLoad)
        self.dataLoadOkButton.setObjectName("dataLoadOkButton")
        self.horizontalLayout.addWidget(self.dataLoadOkButton)
        self.gridLayout.addLayout(self.horizontalLayout, 7, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.line = QtWidgets.QFrame(DataLoad)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.loadDataProgressBar = QtWidgets.QProgressBar(DataLoad)
        self.loadDataProgressBar.setEnabled(False)
        self.loadDataProgressBar.setProperty("value", 0)
        self.loadDataProgressBar.setObjectName("loadDataProgressBar")
        self.verticalLayout.addWidget(self.loadDataProgressBar)

        self.retranslateUi(DataLoad)
        QtCore.QMetaObject.connectSlotsByName(DataLoad)

    def retranslateUi(self, DataLoad):
        _translate = QtCore.QCoreApplication.translate
        DataLoad.setWindowTitle(_translate("DataLoad", "Load Data"))
        self.label_3.setText(_translate("DataLoad", "Parse filename"))
        self.label_4.setText(_translate("DataLoad", "Path inside file"))
        self.label_2.setText(_translate("DataLoad", "Filter Keys"))
        self.fileTypeComboBox.setItemText(0, _translate("DataLoad", "Matlab (.mat)"))
        self.fileTypeComboBox.setItemText(1, _translate("DataLoad", "HDF5 (.h5)"))
        self.label.setText(_translate("DataLoad", "Select datatype"))
        self.pathRecursiveCheckBox.setText(_translate("DataLoad", "Find files recursively"))
        self.pathDataButton.setText(_translate("DataLoad", "Data Path..."))
        self.dataLoadCancelButton.setText(_translate("DataLoad", "Cancel"))
        self.dataLoadOkButton.setText(_translate("DataLoad", "Ok"))
