//! \file PVImportFileDialog.cpp
//! $Id: PVImportFileDialog.cpp 2714 2011-05-12 08:02:27Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QtGui>

#include "PVImportFileDialog.h"

/******************************************************************************
 *
 * PVRush::PVImportFileDialog::PVImportFileDialog
 *
 *****************************************************************************/
PVRush::PVImportFileDialog::PVImportFileDialog(QStringList pluginslist, QWidget *parent) : QFileDialog(parent)
{
	setWindowTitle("Import file");
	setDirectory(QDir().currentPath());
	setFileMode(QFileDialog::ExistingFile);
	treat_as_combobox = new QComboBox();
	QLabel *treat_as_label = new QLabel("Treat file as: ");

	QPushButton *options = new QPushButton("Options");
	options->setMaximumWidth(70);

	options->setCheckable(true);
	options->setAutoDefault(false);

	QGridLayout *this_layout = (QGridLayout *)layout();
	this_layout->addWidget(options, 5, 1);

	QGroupBox *option_group = new QGroupBox();
	option_group->setVisible(0);
	this_layout->addWidget(option_group, 6, 1);

	QGridLayout *options_layout = new QGridLayout();
	//options_layout->setColumnStretch(1, 10);
	option_group->setLayout(options_layout);

	treat_as_combobox->addItems(pluginslist);

	options_layout->addWidget(treat_as_label, 0, 0);
	options_layout->addWidget(treat_as_combobox, 0, 1);

	activate_netflow_checkbox = new QCheckBox("Use Netflow (PCAP only)");
	activate_netflow_checkbox->setChecked(true);

	_check_archives_checkbox = new QCheckBox("Automatically decompress detected archive files");
	_check_archives_checkbox->setChecked(true);

	//options_layout->addWidget(activate_netflow_checkbox, 1, 0);
	//options_layout->addWidget(_check_archives_checkbox, 2, 0);

//	QLabel *read_from_label = new QLabel("Read from line:");
//	options_layout->addWidget(read_from_label, 2, 0);
//	from_line_edit = new QLineEdit("0");
//	options_layout->addWidget(from_line_edit, 2, 1);
//	QLabel *read_to_label = new QLabel(" to ");
//	options_layout->addWidget(read_to_label, 2, 2);
//	to_line_edit = new QLineEdit("0");
//	options_layout->addWidget(to_line_edit, 2, 3);

	setFileMode(QFileDialog::ExistingFiles);

	this->connect(options, SIGNAL(toggled(bool)), option_group, SLOT(setVisible(bool)));
}

/******************************************************************************
 *
 * PVRush::PVImportFileDialog::getFileName()
 *
 *****************************************************************************/
QStringList PVRush::PVImportFileDialog::getFileNames(QString& treat_as)
{
	int         result_dialog_code;
	QStringList list;
	QStringList output_list;

	/* We launch the QFileDialog */
	result_dialog_code = exec();
	/* We check if the user pressed Cancel button */
	if ( result_dialog_code) {
		/* The user didn't press the Cancel button */
		list = selectedFiles();
	}

	treat_as = treat_as_combobox->currentText();

	return list;
}

/******************************************************************************
 *
 * PVRush::PVImportFileDialog::setDefaults()
 *
 *****************************************************************************/
void PVRush::PVImportFileDialog::setDefaults()
{
	treat_as_combobox->setCurrentIndex(0);
	//activate_netflow_checkbox->setChecked(true);
	//from_line_edit->setText(QString("0"));
	//to_line_edit->setText(QString("0"));
	_check_archives_checkbox->setChecked(true);
}
