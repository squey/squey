/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVINSPECTOR_PVAXESCOMBINATIONWIDGET_H
#define PVINSPECTOR_PVAXESCOMBINATIONWIDGET_H

#include <QDialog>
#include <QComboBox>
#include <QList>

#include <inendi/PVAxesCombination.h>

#include <pvguiqt/ui_PVAxesCombinationWidget.h>

namespace Inendi
{
class PVView;
}

namespace PVGuiQt
{

class PVAxesCombinationWidget : public QWidget, Ui::PVAxesCombinationWidget
{
	Q_OBJECT

  public:
	explicit PVAxesCombinationWidget(Inendi::PVAxesCombination& axes_combination,
	                                 Inendi::PVView* view = nullptr,
	                                 QWidget* parent = 0);

  public:
	void reset_used_axes();

  public Q_SLOTS:
	void update_orig_axes();
	void update_used_axes();
	void update_all();

  protected:
	PVCol get_original_axis_selected();
	QString get_original_axis_selected_name();
	QVector<PVCol> get_used_axes_selected();
	bool is_used_axis_selected();
	bool is_original_axis_selected();
	static QVector<PVCol> get_list_selection(QListWidget* widget);
	void set_selection_from_cols(QList<PVCol> const& cols);

  protected Q_SLOTS:
	void axis_add_Slot();
	void axis_up_Slot();
	void axis_down_Slot();
	void axis_remove_Slot();
	void reset_comb_Slot();
	void sel_singleton_Slot();
	void sort_Slot();

  protected:
	Inendi::PVAxesCombination& _axes_combination;
	Inendi::PVView* _view;
};
}

#endif
