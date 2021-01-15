import os
import pandas as pd
import numpy as np
import numbers
from PyQt5 import QtGui, QtCore, QtWidgets


# Compile a QT form
def compile_form(pathToUI):
    pathToPY = os.path.splitext(pathToUI)[0] + ".py"
    qtCompilePrefStr = 'pyuic5 ' + pathToUI + ' -o ' + pathToPY
    print("Compiling QT GUI file", qtCompilePrefStr)
    os.system(qtCompilePrefStr)


def qtable_get_horizontal_header(qtable):
    nColumn = qtable.columnCount()
    return [qtable.horizontalHeaderItem(i).text() for i in range(nColumn)]


def qtable_to_pandas(qtable, selected=False):
    '''
    :param qtable:       QTableWidget object
    :param selected:     Only return selected rows. If False, return whole table
    :return:             Pandas DataFrame corresponding to the QTable
    '''

    nRows = qtable.rowCount()
    nColumns = qtable.columnCount()
    columns = qtable_get_horizontal_header(qtable)

    if selected:
        items = qtable.selectedItems()
        textLst1D = [item.text() for item in items]
        textArr2D = np.array(textLst1D).reshape((len(textLst1D) // nColumns, nColumns))
        return pd.DataFrame(textArr2D, columns=columns)
    else:
        df = pd.DataFrame(columns=columns, index=range(nRows))
        for i in range(nRows):
            for j in range(nColumns):
                df.ix[i, j] = qtable.item(i, j).text()
        return df


def qtable_delete_selected(qtable):
    rowIdxs = [idx.row() for idx in qtable.selectionModel().selectedRows()]
    rowIdxsReverse = sorted(rowIdxs)[::-1]

    for idx in rowIdxsReverse:
        qtable.removeRow(idx)


def qtable_load_from_pandas(qtable, df, append=False):
    if not append:
        # self.gui.collectionCardsTable.clear()
        qtable.setRowCount(0)

        # Set columns of QTable as DataFrame columns
        qtable.setColumnCount(len(df.columns))
        qtable.setHorizontalHeaderLabels(list(df.columns))

    for idx, row in df.iterrows():
        rowIdxQtable = qtable.rowCount()

        qtable.insertRow(rowIdxQtable)
        for iCol, cell in enumerate(list(row)):
            if isinstance(cell, numbers.Number):
                item = QtWidgets.QTableWidgetItem()
                item.setData(QtCore.Qt.DisplayRole, cell)
            else:
                item = QtWidgets.QTableWidgetItem(str(cell))
            qtable.setItem(rowIdxQtable, iCol, item)

    qtable.resizeColumnsToContents()


def qtmenu_add_item(menu, name, icon=None, shortcut=None, statusTip=None, actionFunc=None):
    qaction = QtWidgets.QAction(icon, '&' + name, menu)
    if shortcut is not None:
        qaction.setShortcut(shortcut)

    if statusTip is not None:
        qaction.setStatusTip(statusTip)

    if qaction is not None:
        qaction.triggered.connect(actionFunc)

    menu.addAction(qaction)
