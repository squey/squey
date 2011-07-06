//! \file PVExportSelectionDialog.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

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
