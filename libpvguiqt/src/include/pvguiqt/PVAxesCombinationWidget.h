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
} // namespace Inendi

namespace PVGuiQt
{

class PVAxesCombinationWidget : public QWidget, Ui::PVAxesCombinationWidget
{
	Q_OBJECT

  public:
	explicit PVAxesCombinationWidget(Inendi::PVAxesCombination& axes_combination,
	                                 Inendi::PVView* view = nullptr,
	                                 QWidget* parent = nullptr);

  public:
	void reset_used_axes();

  public Q_SLOTS:
	void update_orig_axes();
	void update_used_axes();
	void update_all();

  protected:
	QVector<PVCol> get_used_axes_selected();
	bool is_used_axis_selected();
	bool is_original_axis_selected();
	static QVector<PVCol> get_list_selection(QListWidget* widget);
	void set_selection_from_cols(QList<PVCol> const& cols);
	void update_combination();
	void enable_drop(bool enable);
	QList<QListWidgetItem*> ordered_selected(QListWidget* list_widget) const;

  protected Q_SLOTS:
	void reset_comb_Slot();
	void axis_add_before_Slot();
	void axis_add_after_Slot();
	void select_all_Slot();
	void invert_selection_Slot();
	void axis_up_Slot();
	void axis_down_Slot();
	void move_top_Slot();
	void move_bottom_Slot();
	void gather_selected_Slot();
	void axis_remove_Slot();
	void remove_duplicates_Slot();
	void sort_Slot();
	void sel_singleton_Slot();

  protected:
	Inendi::PVAxesCombination& _axes_combination;
	Inendi::PVView* _view;
	bool _dnd_enabled = false;
	QMetaObject::Connection _connection_dnd_inserted;
	QMetaObject::Connection _connection_dnd_moved;
	QMetaObject::Connection _connection_dnd_removed;

	struct DisableDnD {
		PVAxesCombinationWidget* const parent;
		DisableDnD(auto* parent) : parent(parent) { parent->enable_drop(false); }
		~DisableDnD() { parent->enable_drop(true); }
		DisableDnD(DisableDnD&&) = delete;
	};
};
} // namespace PVGuiQt

#endif
