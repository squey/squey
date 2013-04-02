/**
 * \file PVNewLayerDialog.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#include <picviz/widgets/PVNewLayerDialog.h>

#include <QDialogButtonBox>
#include <QLabel>
#include <QVBoxLayout>

PVWidgets::PVNewLayerDialog::PVNewLayerDialog(const QString& layer_name, bool hide_layers, QWidget* parent /*= 0*/) : QDialog(parent)
{
	QVBoxLayout* layout = new QVBoxLayout();
	QLabel* label = new QLabel("Layer name:");
	_text = new QLineEdit(layer_name);
	_text->setSelection(0, layer_name.length());
	label->setBuddy(_text);
	_checkbox = new QCheckBox("Hide all other layers");
	_checkbox->setChecked(hide_layers);

	QDialogButtonBox* button_box = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

	connect(button_box, SIGNAL(accepted()), this, SLOT(accept()));
	connect(button_box, SIGNAL(rejected()), this, SLOT(reject()));

	layout->addWidget(label);
	layout->addWidget(_text);
	layout->addWidget(_checkbox);
	layout->addWidget(button_box);

	setLayout(layout);
}

bool PVWidgets::PVNewLayerDialog::should_hide_layers() const
{
	return _checkbox->checkState() == Qt::Checked;
}

QString PVWidgets::PVNewLayerDialog::get_layer_name() const
{
	return _text->text();
}

QString PVWidgets::PVNewLayerDialog::get_new_layer_name_from_dialog(const QString& layer_name, bool& hide_layers)
{
	PVNewLayerDialog* dialog = new PVNewLayerDialog(layer_name, hide_layers);
	int res = dialog->exec();

	dialog->deleteLater();

	hide_layers = dialog->should_hide_layers();
	return (res == QDialog::Accepted) ? dialog->get_layer_name() : QString();
}

