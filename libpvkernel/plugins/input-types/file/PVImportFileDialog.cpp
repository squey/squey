/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <QComboBox>
#include <QFormLayout>
#include <QGroupBox>

#include "PVImportFileDialog.h"

/******************************************************************************
 *
 * PVRush::PVImportFileDialog::PVImportFileDialog
 *
 *****************************************************************************/
PVRush::PVImportFileDialog::PVImportFileDialog(QStringList pluginslist, QWidget* parent)
    : PVWidgets::PVFileDialog(parent)
{
	setWindowTitle("Import file");
	setFileMode(QFileDialog::ExistingFiles);

	QGridLayout* this_layout = (QGridLayout*)layout();

	QGroupBox* option_group = new QGroupBox();
	this_layout->addWidget(option_group, 6, 0, 1, 3);

	QFormLayout* form_layout = new QFormLayout();
	option_group->setLayout(form_layout);

	treat_as_combobox = new QComboBox();
	treat_as_combobox->addItems(pluginslist);

	form_layout->addRow(tr("Format: "), treat_as_combobox);
}

/******************************************************************************
 *
 * PVRush::PVImportFileDialog::getFileName()
 *
 *****************************************************************************/
QStringList PVRush::PVImportFileDialog::getFileNames(QString& treat_as)
{

	/* Launch the Dialog and check if the user pressed Cancel button */
	if (not exec()) {
		return {};
	}

	/* The user didn't press the Cancel button */
	treat_as = treat_as_combobox->currentText();

	return selectedFiles();
}
