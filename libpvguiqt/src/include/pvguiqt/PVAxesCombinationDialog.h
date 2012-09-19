/**
 * \file PVAxesCombinationDialog.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVGUIQT_PVAXESCOMBINATIONDIALOG_H
#define PVGUIQT_PVAXESCOMBINATIONDIALOG_H

#include <pvkernel/core/general.h>

#include <picviz/PVAxesCombination.h>
#include <picviz/PVView_types.h>

#include <pvhive/PVActor.h>
#include <pvhive/PVObserverSignal.h>

#include <QDialog>
#include <QDialogButtonBox>
#include <QMessageBox>

namespace PVGuiQt {

class PVAxesCombinationWidget;

class PVAxesCombinationDialog: public QDialog
{
	Q_OBJECT

public:
	PVAxesCombinationDialog(Picviz::PVView_sp& view, QWidget* parent = NULL);
	~PVAxesCombinationDialog();

public:
	void save_current_combination();
	void update_used_axes();

private:
	Picviz::PVView const& lib_view() const { return _lib_view; }

protected slots:
	void axes_comb_updated();
	void view_about_to_be_deleted();
	void commit_axes_comb_to_view();
	void box_btn_clicked(QAbstractButton* btn);
	void update_box_answered(QAbstractButton* btn);

protected:
	PVAxesCombinationWidget* _axes_widget;
	PVHive::PVObserverSignal<Picviz::PVAxesCombination::columns_indexes_t> _obs_axes_comb;
	PVHive::PVActor<Picviz::PVView> _actor;
	Picviz::PVAxesCombination _temp_axes_comb;
	Picviz::PVView const& _lib_view;
	bool _valid;

private:
	QDialogButtonBox* _box_buttons;
	QMessageBox* _update_box;
};

}

#endif
