# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CalcTransformsUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_CalcTransformsUI(object):
    def setupUi(self, CalcTransformsUI):
        CalcTransformsUI.setObjectName("CalcTransformsUI")
        CalcTransformsUI.resize(412, 218)
        self.verticalLayout = QtWidgets.QVBoxLayout(CalcTransformsUI)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(CalcTransformsUI)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.location_Text = QtWidgets.QLineEdit(CalcTransformsUI)
        self.location_Text.setEnabled(False)
        self.location_Text.setObjectName("location_Text")
        self.horizontalLayout.addWidget(self.location_Text)
        self.choose_Button = QtWidgets.QPushButton(CalcTransformsUI)
        self.choose_Button.setObjectName("choose_Button")
        self.horizontalLayout.addWidget(self.choose_Button)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.progressBar = QtWidgets.QProgressBar(CalcTransformsUI)
        self.progressBar.setEnabled(False)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout.addWidget(self.progressBar)
        self.orientationBox = QtWidgets.QGroupBox(CalcTransformsUI)
        self.orientationBox.setObjectName("orientationBox")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.orientationBox)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.original_radio = QtWidgets.QRadioButton(self.orientationBox)
        self.original_radio.setChecked(True)
        self.original_radio.setObjectName("original_radio")
        self.horizontalLayout_2.addWidget(self.original_radio)
        self.axial_radio = QtWidgets.QRadioButton(self.orientationBox)
        self.axial_radio.setObjectName("axial_radio")
        self.horizontalLayout_2.addWidget(self.axial_radio)
        self.sagittal_radio = QtWidgets.QRadioButton(self.orientationBox)
        self.sagittal_radio.setObjectName("sagittal_radio")
        self.horizontalLayout_2.addWidget(self.sagittal_radio)
        self.coronal_radio = QtWidgets.QRadioButton(self.orientationBox)
        self.coronal_radio.setObjectName("coronal_radio")
        self.horizontalLayout_2.addWidget(self.coronal_radio)
        self.verticalLayout.addWidget(self.orientationBox)
        spacerItem = QtWidgets.QSpacerItem(20, 45, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.calculate_button = QtWidgets.QPushButton(CalcTransformsUI)
        self.calculate_button.setEnabled(False)
        self.calculate_button.setObjectName("calculate_button")
        self.verticalLayout.addWidget(self.calculate_button)

        self.retranslateUi(CalcTransformsUI)
        QtCore.QMetaObject.connectSlotsByName(CalcTransformsUI)

    def retranslateUi(self, CalcTransformsUI):
        _translate = QtCore.QCoreApplication.translate
        CalcTransformsUI.setWindowTitle(_translate("CalcTransformsUI", "Form"))
        self.label.setText(_translate("CalcTransformsUI", "Location:"))
        self.choose_Button.setText(_translate("CalcTransformsUI", "Choose..."))
        self.orientationBox.setTitle(_translate("CalcTransformsUI", "Orientation"))
        self.original_radio.setText(_translate("CalcTransformsUI", "Original"))
        self.axial_radio.setText(_translate("CalcTransformsUI", "Axial"))
        self.sagittal_radio.setText(_translate("CalcTransformsUI", "Sagittal"))
        self.coronal_radio.setText(_translate("CalcTransformsUI", "Coronal"))
        self.calculate_button.setText(_translate("CalcTransformsUI", "Calculate Transforms"))
