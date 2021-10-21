# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/media/aleksejs/DataHDD/work/codes/comp-neuro/analysis-mesoscopic/mesostat-dev/mesostat/gui/importh5.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_importH5Dialog(object):
    def setupUi(self, importH5Dialog):
        importH5Dialog.setObjectName("importH5Dialog")
        importH5Dialog.resize(400, 300)
        self.gridLayout = QtWidgets.QGridLayout(importH5Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.extraParamsLineEdit = QtWidgets.QLineEdit(importH5Dialog)
        self.extraParamsLineEdit.setObjectName("extraParamsLineEdit")
        self.gridLayout.addWidget(self.extraParamsLineEdit, 3, 1, 1, 1)
        self.newDatasetRadioButton = QtWidgets.QRadioButton(importH5Dialog)
        self.newDatasetRadioButton.setObjectName("newDatasetRadioButton")
        self.gridLayout.addWidget(self.newDatasetRadioButton, 0, 0, 1, 1)
        self.existingDatasetComboBox = QtWidgets.QComboBox(importH5Dialog)
        self.existingDatasetComboBox.setObjectName("existingDatasetComboBox")
        self.gridLayout.addWidget(self.existingDatasetComboBox, 1, 1, 1, 1)
        self.existingDatasetRadioButtion = QtWidgets.QRadioButton(importH5Dialog)
        self.existingDatasetRadioButtion.setObjectName("existingDatasetRadioButtion")
        self.gridLayout.addWidget(self.existingDatasetRadioButtion, 1, 0, 1, 1)
        self.parseLabelLineEdit = QtWidgets.QLineEdit(importH5Dialog)
        self.parseLabelLineEdit.setObjectName("parseLabelLineEdit")
        self.gridLayout.addWidget(self.parseLabelLineEdit, 2, 1, 1, 1)
        self.extraParamsLabel = QtWidgets.QLabel(importH5Dialog)
        self.extraParamsLabel.setObjectName("extraParamsLabel")
        self.gridLayout.addWidget(self.extraParamsLabel, 3, 0, 1, 1)
        self.newDatasetLineEdit = QtWidgets.QLineEdit(importH5Dialog)
        self.newDatasetLineEdit.setObjectName("newDatasetLineEdit")
        self.gridLayout.addWidget(self.newDatasetLineEdit, 0, 1, 1, 1)
        self.parseLabelCheckBox = QtWidgets.QCheckBox(importH5Dialog)
        self.parseLabelCheckBox.setObjectName("parseLabelCheckBox")
        self.gridLayout.addWidget(self.parseLabelCheckBox, 2, 0, 1, 1)
        self.cancelPushButton = QtWidgets.QPushButton(importH5Dialog)
        self.cancelPushButton.setObjectName("cancelPushButton")
        self.gridLayout.addWidget(self.cancelPushButton, 4, 0, 1, 1)
        self.okPushButton = QtWidgets.QPushButton(importH5Dialog)
        self.okPushButton.setObjectName("okPushButton")
        self.gridLayout.addWidget(self.okPushButton, 4, 1, 1, 1)

        self.retranslateUi(importH5Dialog)
        QtCore.QMetaObject.connectSlotsByName(importH5Dialog)

    def retranslateUi(self, importH5Dialog):
        _translate = QtCore.QCoreApplication.translate
        importH5Dialog.setWindowTitle(_translate("importH5Dialog", "Dialog"))
        self.newDatasetRadioButton.setText(_translate("importH5Dialog", "As new dataset"))
        self.existingDatasetRadioButtion.setText(_translate("importH5Dialog", "To existing dataset"))
        self.extraParamsLabel.setText(_translate("importH5Dialog", "Extra params"))
        self.parseLabelCheckBox.setText(_translate("importH5Dialog", "Parse label"))
        self.cancelPushButton.setText(_translate("importH5Dialog", "Cancel"))
        self.okPushButton.setText(_translate("importH5Dialog", "Ok"))

