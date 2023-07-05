/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVSQUEY_PVAXESCOMBINATIONWIDGET_H
#define PVSQUEY_PVAXESCOMBINATIONWIDGET_H

#include <QDialog>
#include <QComboBox>
#include <QList>

#include <squey/PVAxesCombination.h>

#include "pvguiqt/ui_PVAxesCombinationWidget.h"
namespace Squey
{
class PVView;
} // namespace Squey

namespace PVGuiQt
{

class PVAxesCombinationWidget : public QWidget, Ui::PVAxesCombinationWidget
{
	Q_OBJECT

  public:
	explicit PVAxesCombinationWidget(Squey::PVAxesCombination& axes_combination,
	                                 Squey::PVView* view = nullptr,
	                                 QWidget* parent = nullptr);

  public:
	void reset_used_axes();

  Q_SIGNALS:
	void closed();

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
	void closeEvent(QCloseEvent *event) override;

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
	Squey::PVAxesCombination& _axes_combination;
	Squey::PVView* _view;
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
