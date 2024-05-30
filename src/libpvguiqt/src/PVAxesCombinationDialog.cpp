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

#include <squey/PVView.h>

#include <pvguiqt/PVAxesCombinationDialog.h>
#include <pvguiqt/PVAxesCombinationWidget.h>

#include <QVBoxLayout>
#include <QPushButton>

PVGuiQt::PVAxesCombinationDialog::PVAxesCombinationDialog(Squey::PVView& view, QWidget* parent)
    : QDialog(parent), _temp_axes_comb(view.get_axes_combination()), _lib_view(view)
{
	auto* main_layout = new QVBoxLayout(this);
	_box_buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel |
	                                    QDialogButtonBox::Apply);
	_axes_widget = new PVAxesCombinationWidget(_temp_axes_comb, &view);
	main_layout->addWidget(_axes_widget);
	main_layout->addWidget(_box_buttons);
	setLayout(main_layout);

	// Buttons
	connect(_box_buttons, &QDialogButtonBox::accepted, this,
	        &PVAxesCombinationDialog::commit_axes_comb_to_view);
	connect(_box_buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(_box_buttons, &QDialogButtonBox::clicked, this,
	        &PVAxesCombinationDialog::box_btn_clicked);
	connect(_box_buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

	setWindowTitle("Axes combination");
}

void PVGuiQt::PVAxesCombinationDialog::reset_used_axes()
{
	_axes_widget->reset_used_axes();
}

void PVGuiQt::PVAxesCombinationDialog::commit_axes_comb_to_view()
{
	lib_view().set_axes_combination(_temp_axes_comb.get_combination());
}

void PVGuiQt::PVAxesCombinationDialog::box_btn_clicked(QAbstractButton* btn)
{
	if (btn == static_cast<QAbstractButton*>(_box_buttons->button(QDialogButtonBox::Apply))) {
		commit_axes_comb_to_view();
	}
}
