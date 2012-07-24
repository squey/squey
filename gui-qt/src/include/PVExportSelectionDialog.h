/**
 * \file PVExportSelectionDialog.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVEXPORTSELECTIONDIALOG_H
#define PVEXPORTSELECTIONDIALOG_H

#include <QFileDialog>

namespace PVInspector {

/**
 * \class PVExportSelectionDialog
 */
class PVExportSelectionDialog : public QFileDialog
{
    Q_OBJECT

public:
	PVExportSelectionDialog(QWidget *parent = 0);

};
}

#endif // PVEXPORTSELECTIONDIALOG_H
