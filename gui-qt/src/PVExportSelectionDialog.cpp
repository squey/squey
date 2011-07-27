//! \file PVExportSelectionDialog.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

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
