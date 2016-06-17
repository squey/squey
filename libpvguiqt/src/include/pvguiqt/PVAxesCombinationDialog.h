/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVGUIQT_PVAXESCOMBINATIONDIALOG_H
#define PVGUIQT_PVAXESCOMBINATIONDIALOG_H

#include <inendi/PVAxesCombination.h>

#include <QDialog>
#include <QDialogButtonBox>

namespace PVGuiQt
{

class PVAxesCombinationWidget;

class PVAxesCombinationDialog : public QDialog
{
	Q_OBJECT

  public:
	PVAxesCombinationDialog(Inendi::PVView_sp& view, QWidget* parent = nullptr);

  public:
	void reset_used_axes();

  private:
	Inendi::PVView& lib_view() { return _lib_view; }

  protected Q_SLOTS:
	void commit_axes_comb_to_view();
	void box_btn_clicked(QAbstractButton* btn);

  protected:
	PVAxesCombinationWidget* _axes_widget;
	Inendi::PVAxesCombination _temp_axes_comb;
	Inendi::PVView& _lib_view;

  private:
	QDialogButtonBox* _box_buttons;
};
}

#endif
