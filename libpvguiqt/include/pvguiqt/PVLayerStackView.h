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

#ifndef PVLAYERSTACKVIEW_H
#define PVLAYERSTACKVIEW_H

#include <QTableView>

#include <functional>

#include <pvkernel/widgets/PVFileDialog.h>

namespace Inendi
{
class PVLayer;
class PVSelection;
} // namespace Inendi

namespace PVGuiQt
{

class PVLayerStackModel;

/**
 *  \class PVLayerStackView
 */
class PVLayerStackView : public QTableView
{
	Q_OBJECT

  public:
	explicit PVLayerStackView(QWidget* parent = nullptr);

  public:
	PVLayerStackModel* ls_model();

  private:
	using operation_f =
	    Inendi::PVSelection (Inendi::PVSelection::*)(const Inendi::PVSelection&) const;

  private:
	void copy_to_clipboard();
	void set_current_selection_from_layer(int model_idx);
	void export_layer_selection(int model_idx);
	void reset_layer_colors(int layer_idx);
	void show_this_layer_only(int layer_idx);
	void
	boolean_op_on_selection_with_this_layer(int layer_idx, const operation_f& f, bool activate);

  public:
	Inendi::PVLayer& get_layer_from_idx(int model_idx);

  private Q_SLOTS:
	void show_ctxt_menu(QPoint const& pt);

  protected:
	void enterEvent(QEnterEvent* event) override;
	void leaveEvent(QEvent* event) override;
	void mouseDoubleClickEvent(QMouseEvent* event) override;
	void keyPressEvent(QKeyEvent* event) override;

  private:
	// Context menu
	QMenu* _ctxt_menu;
	QAction* _ctxt_menu_copy_to_clipboard_act;
	QAction* _ctxt_menu_set_sel_layer;
	QAction* _ctxt_menu_export_layer_sel;
	QAction* _ctxt_menu_reset_colors;
	QAction* _ctxt_menu_show_this_layer_only;

	QAction* _ctxt_menu_union;
	QAction* _ctxt_menu_intersection;
	QAction* _ctxt_menu_difference;
	QAction* _ctxt_menu_symmetric_differrence;

	QAction* _ctxt_menu_activate_union;
	QAction* _ctxt_menu_activate_difference;
	QAction* _ctxt_menu_activate_intersection;
	QAction* _ctxt_menu_activate_symmetric_differrence;

	PVWidgets::PVFileDialog _layer_dialog;
	PVWidgets::PVFileDialog _layerstack_dialog;
};
} // namespace PVGuiQt

#endif // PVLAYERSTACKVIEW_H
