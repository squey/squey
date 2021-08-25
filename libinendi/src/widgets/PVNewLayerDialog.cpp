//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

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

	connect(button_box, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(button_box, &QDialogButtonBox::rejected, this, &QDialog::reject);

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
