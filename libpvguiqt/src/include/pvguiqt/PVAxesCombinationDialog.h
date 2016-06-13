/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVGUIQT_PVAXESCOMBINATIONDIALOG_H
#define PVGUIQT_PVAXESCOMBINATIONDIALOG_H

#include <inendi/PVAxesCombination.h>
#include <inendi/PVView_types.h>

#include <pvhive/PVActor.h>
#include <pvhive/PVObserverSignal.h>

#include <QDialog>
#include <QDialogButtonBox>
#include <QMessageBox>

namespace PVGuiQt
{

class PVAxesCombinationWidget;

class PVAxesCombinationDialog : public QDialog
{
	Q_OBJECT

  public:
	PVAxesCombinationDialog(Inendi::PVView_sp& view, QWidget* parent = NULL);
	~PVAxesCombinationDialog();

  public:
	void update_used_axes();

	void reset_used_axes();

  private:
	Inendi::PVView const& lib_view() const { return _lib_view; }

  protected slots:
	void axes_comb_updated();
	void view_about_to_be_deleted();
	void commit_axes_comb_to_view();
	void box_btn_clicked(QAbstractButton* btn);
	void update_box_answered(QAbstractButton* btn);

  protected:
	PVAxesCombinationWidget* _axes_widget;
	PVHive::PVObserverSignal<Inendi::PVAxesCombination::columns_indexes_t> _obs_axes_comb;
	PVHive::PVActor<Inendi::PVView> _actor;
	Inendi::PVAxesCombination _temp_axes_comb;
	Inendi::PVView const& _lib_view;
	bool _valid;

  private:
	QDialogButtonBox* _box_buttons;
	QMessageBox* _update_box;
};
}

#endif
