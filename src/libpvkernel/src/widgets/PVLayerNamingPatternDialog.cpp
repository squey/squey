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

#include <pvkernel/widgets/PVLayerNamingPatternDialog.h>
#include <QtCore/qobjectdefs.h>
#include <QLineEdit>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QLabel>
#include <QVBoxLayout>

class QWidget;

/******************************************************************************
 * PVWidgets::PVLayerNamingPatternDialog::PVLayerNamingPatternDialog
 *****************************************************************************/

PVWidgets::PVLayerNamingPatternDialog::PVLayerNamingPatternDialog(const QString& title,
                                                                  const QString& text,
                                                                  const QString& pattern,
                                                                  insert_mode m,
                                                                  QWidget* parent)
    : QDialog(parent)
{
	setWindowTitle(title);

	auto vlayout = new QVBoxLayout();
	setLayout(vlayout);

	// the explaination text
	auto* label = new QLabel(text + "; substitution form:\n%l: current layer's name\n%a: axis' "
	                                  "name\n%v: comma separated values");
	vlayout->addWidget(label);

	// the pattern edit
	_line_edit = new QLineEdit(pattern);
	vlayout->addWidget(_line_edit);

	// the placement choice
	auto hlayout = new QHBoxLayout();
	vlayout->addLayout(hlayout);

	label = new QLabel("placement:");
	hlayout->addWidget(label);

	_combo_box = new QComboBox();
	_combo_box->addItem("On top of the layer stack");
	_combo_box->addItem("Above the current layer");
	_combo_box->addItem("Below the current layer");
	_combo_box->setCurrentIndex(m);
	hlayout->addWidget(_combo_box);

	// the button box
	auto button_box = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	connect(button_box, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(button_box, &QDialogButtonBox::rejected, this, &QDialog::reject);

	vlayout->addWidget(button_box);
}

/******************************************************************************
 * PVWidgets::PVLayerNamingPatternDialog::get_name_pattern
 *****************************************************************************/

QString PVWidgets::PVLayerNamingPatternDialog::get_name_pattern() const
{
	return _line_edit->text();
}

/******************************************************************************
 * PVWidgets::PVLayerNamingPatternDialog::get_insertion_mode
 *****************************************************************************/

PVWidgets::PVLayerNamingPatternDialog::insert_mode
PVWidgets::PVLayerNamingPatternDialog::get_insertion_mode() const
{
	return (insert_mode)_combo_box->currentIndex();
}
