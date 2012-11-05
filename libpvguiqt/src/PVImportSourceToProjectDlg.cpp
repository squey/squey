/**
 * \file PVImportSourceToProjectDlg.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvguiqt/PVImportSourceToProjectDlg.h>
#include <QDialogButtonBox>
#include <QBoxLayout>
#include <QLabel>
#include <QComboBox>

PVGuiQt::PVImportSourceToProjectDlg::PVImportSourceToProjectDlg(const QStringList & list, int default_index, QWidget* parent /* = 0 */) :
	QDialog(parent)
{
	setWindowTitle(tr("Select project"));

	QVBoxLayout* layout = new QVBoxLayout();

	QHBoxLayout* hbox_layout = new QHBoxLayout();
	QLabel* label = new QLabel(tr("Import source to project:"));

	QComboBox* combo_box = new QComboBox();
	connect(combo_box, SIGNAL(currentIndexChanged(int)), this, SLOT(setResult(int)));
	combo_box->addItems(list);
	combo_box->setCurrentIndex(default_index);

	hbox_layout->addWidget(label);
	hbox_layout->addWidget(combo_box);

	QDialogButtonBox* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(buttons, SIGNAL(accepted()), this, SLOT(accept()));
    connect(buttons, SIGNAL(rejected()), this, SLOT(reject()));

    layout->addLayout(hbox_layout);
	layout->addWidget(buttons);

	setLayout(layout);

	show();
}
