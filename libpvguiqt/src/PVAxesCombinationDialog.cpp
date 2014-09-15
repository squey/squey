/**
 * \file PVAxesCombinationDialog.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvhive/PVHive.h>

#include <pvguiqt/PVAxesCombinationDialog.h>
#include <pvguiqt/PVAxesCombinationWidget.h>

#include <QVBoxLayout>

PVGuiQt::PVAxesCombinationDialog::PVAxesCombinationDialog(Picviz::PVView_sp& view, QWidget* parent):
	QDialog(parent),
	_lib_view(*view),
	_valid(true)
{
	QVBoxLayout* main_layout = new QVBoxLayout();
	_box_buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel | QDialogButtonBox::Apply);
	_temp_axes_comb = view->get_axes_combination();
	_axes_widget = new PVAxesCombinationWidget(_temp_axes_comb, view.get());
	main_layout->addWidget(_axes_widget);
	main_layout->addWidget(_box_buttons);
	setLayout(main_layout);

	_update_box = new QMessageBox(QMessageBox::Question, tr("Axes combination modified..."), tr("The current axes combination has been modified by an external window. Do you want to erase your current combination and start from the new one ?"), QMessageBox::Yes | QMessageBox::No, this);
	_update_box->setModal(false);
	connect(_update_box, SIGNAL(buttonClicked(QAbstractButton*)), this, SLOT(update_box_answered(QAbstractButton*)));

	// Buttons
	connect(_box_buttons, SIGNAL(accepted()), this, SLOT(commit_axes_comb_to_view()));
	connect(_box_buttons, SIGNAL(accepted()), this, SLOT(accept()));
	connect(_box_buttons, SIGNAL(clicked(QAbstractButton*)),  this, SLOT(box_btn_clicked(QAbstractButton*)));
	connect(_box_buttons, SIGNAL(rejected()), this, SLOT(reject()));

	// Hive
	_obs_axes_comb.connect_refresh(_update_box, SLOT(show()));
	_obs_axes_comb.connect_about_to_be_deleted(this, SLOT(view_about_to_be_deleted()));
	PVHive::get().register_observer(view, [=](Picviz::PVView& v) { return &v.get_axes_combination().get_axes_index_list(); }, _obs_axes_comb);
	PVHive::get().register_actor(view, _actor);


	setWindowTitle("Edit axes combination... [" + view->get_name() + "]");
}

PVGuiQt::PVAxesCombinationDialog::~PVAxesCombinationDialog()
{
	_axes_widget->deleteLater();
	_update_box->deleteLater();
}

void PVGuiQt::PVAxesCombinationDialog::reset_used_axes()
{
	_axes_widget->reset_used_axes();
}

void PVGuiQt::PVAxesCombinationDialog::commit_axes_comb_to_view()
{
	if (_valid) {
		_obs_axes_comb.disconnect_refresh(_update_box, SLOT(show()));
		_actor.call<FUNC(Picviz::PVView::set_axes_combination_list_id)>(_temp_axes_comb.get_axes_index_list(), _temp_axes_comb.get_axes_list());
		_obs_axes_comb.connect_refresh(_update_box, SLOT(show()));
	}
}

void PVGuiQt::PVAxesCombinationDialog::axes_comb_updated()
{
	_update_box->show();
}

void PVGuiQt::PVAxesCombinationDialog::view_about_to_be_deleted()
{
	_valid = false;
	_update_box->hide();
	reject();
}

void PVGuiQt::PVAxesCombinationDialog::box_btn_clicked(QAbstractButton* btn)
{
	if (btn == static_cast<QAbstractButton*>(_box_buttons->button(QDialogButtonBox::Apply))) {
		commit_axes_comb_to_view();
	}
}

void PVGuiQt::PVAxesCombinationDialog::update_box_answered(QAbstractButton* btn)
{
	if (btn == static_cast<QAbstractButton*>(_update_box->button(QMessageBox::Yes))) {
		_temp_axes_comb = lib_view().get_axes_combination();
		_axes_widget->update_used_axes();
	}
}
