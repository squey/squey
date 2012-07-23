/**
 * \file PVExportSelectionDialog.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <PVExportSelectionDialog.h>

/******************************************************************************
 *
 * PVInspector::PVExportSelectionDialog::PVExportSelectionDialog
 *
 *****************************************************************************/
PVInspector::PVExportSelectionDialog::PVExportSelectionDialog(QWidget *parent) :
	QFileDialog(parent, tr("Export selection as..."))
{
	setAcceptMode(QFileDialog::AcceptSave);
}
