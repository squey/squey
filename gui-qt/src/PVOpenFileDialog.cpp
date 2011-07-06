//! \file PVOpenFileDialog.cpp
//! $Id: PVOpenFileDialog.cpp 2496 2011-04-25 14:10:00Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QtGui>

#include <PVOpenFileDialog.h>

#include <picviz/general.h>
#include <picviz/plugins.h>
//#include <picviz/open-save.h>

/******************************************************************************
 *
 * PVInspector::PVOpenFileDialog::PVOpenFileDialog
 *
 *****************************************************************************/
PVInspector::PVOpenFileDialog::PVOpenFileDialog(QWidget *parent) : QFileDialog(parent)
{
	setWindowTitle("Open file");
	setDirectory(QDir().currentPath());
	setFileMode(QFileDialog::ExistingFile);
}



/******************************************************************************
 *
 * PVInspector::PVOpenFileDialog::getFileName()
 *
 *****************************************************************************/
QStringList PVInspector::PVOpenFileDialog::getFileName()
{
	/* VARIABLES */
	int result_dialog_code;
	QString test;
	QString file_absolute_path;
	QStringList list;
	QStringList output_list;

	/* CODE */
	/* We launch the QFileDialog */
	result_dialog_code = exec();
	/* We check if the user pressed Cancel button */
	if ( result_dialog_code) {
		/* The user didn't press the Cancel button */
		list = selectedFiles();
		file_absolute_path = list.first();
	} else {
		/* The user did cancel the dialog */
		file_absolute_path = "";
	}

	/* We fill the output_list with the two values */
	output_list << file_absolute_path;

	return output_list;
}















