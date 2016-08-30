/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVView.h>

#include <pvguiqt/PVAxesCombinationDialog.h>
#include <pvguiqt/PVAxesCombinationWidget.h>

#include <QVBoxLayout>

PVGuiQt::PVAxesCombinationDialog::PVAxesCombinationDialog(Inendi::PVView& view, QWidget* parent)
    : QDialog(parent), _temp_axes_comb(view.get_axes_combination()), _lib_view(view)
{
	QVBoxLayout* main_layout = new QVBoxLayout(this);
	_box_buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel |
	                                    QDialogButtonBox::Apply);
	_axes_widget = new PVAxesCombinationWidget(_temp_axes_comb, &view);
	main_layout->addWidget(_axes_widget);
	main_layout->addWidget(_box_buttons);
	setLayout(main_layout);

	// Buttons
	connect(_box_buttons, SIGNAL(accepted()), this, SLOT(commit_axes_comb_to_view()));
	connect(_box_buttons, SIGNAL(accepted()), this, SLOT(accept()));
	connect(_box_buttons, SIGNAL(clicked(QAbstractButton*)), this,
	        SLOT(box_btn_clicked(QAbstractButton*)));
	connect(_box_buttons, SIGNAL(rejected()), this, SLOT(reject()));

	setWindowTitle("Edit axes combination... [" + QString::fromStdString(view.get_name()) + "]");
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
