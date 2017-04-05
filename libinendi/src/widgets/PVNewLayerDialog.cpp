/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/widgets/PVNewLayerDialog.h>

#include <QDialogButtonBox>
#include <QLabel>
#include <QVBoxLayout>
#include <QApplication>
#include <QCursor>

PVWidgets::PVNewLayerDialog::PVNewLayerDialog(const QString& layer_name,
                                              bool hide_layers,
                                              QWidget* parent /*= 0*/)
    : QDialog(parent)
{
	auto layout = new QVBoxLayout();
	QLabel* label = new QLabel("Layer name:");
	_text = new QLineEdit(layer_name);
	_text->setSelection(0, layer_name.length());
	label->setBuddy(_text);
	_checkbox = new QCheckBox("Hide all other layers");
	_checkbox->setChecked(hide_layers);

	auto button_box = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

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

QString PVWidgets::PVNewLayerDialog::get_new_layer_name_from_dialog(const QString& layer_name,
                                                                    bool& hide_layers,
                                                                    QWidget* parent_widget)
{
	QWidget* w = QApplication::widgetAt(QCursor::pos());

	if (w == nullptr) {
		/* We make sure this dialog can not be hidden under another window by giving it a
		 * non-null parent. As no Inspector window (PVMainwindow, undocked graphic view,
		 * etc.) has been found under the mouse cursor, we use the one invoking this method.
		 */
		w = parent_widget;
	}
	auto dialog = new PVNewLayerDialog(layer_name, hide_layers, w);
	int res = dialog->exec();

	dialog->deleteLater();

	hide_layers = dialog->should_hide_layers();
	return (res == QDialog::Accepted) ? dialog->get_layer_name() : QString();
}
