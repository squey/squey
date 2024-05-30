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

#include <QAction>
#include <QToolBar>
#include <QVBoxLayout>
#include <QHeaderView>

#include <pvguiqt/PVLayerStackDelegate.h>
#include <pvguiqt/PVLayerStackModel.h>
#include <pvguiqt/PVLayerStackView.h>
#include <pvguiqt/PVLayerStackWidget.h>
#include <pvguiqt/PVExportSelectionDlg.h>
#include <pvkernel/widgets/PVModdedIcon.h>

#include <squey/widgets/PVNewLayerDialog.h>

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackWidget::PVLayerStackWidget
 *
 *****************************************************************************/
PVGuiQt::PVLayerStackWidget::PVLayerStackWidget(Squey::PVView& lib_view, QWidget* parent)
    : QWidget(parent)
{
	QVBoxLayout* main_layout;
	QToolBar* layer_stack_toolbar;

	// SIZE STUFF
	// WARNING: nothing should be set here.

	// OBJECTNAME STUFF
	setObjectName("PVLayerStackWidget");

	// LAYOUT STUFF
	// We need a Layout for that Widget
	main_layout = new QVBoxLayout(this);
	// We fix the margins for that Layout
	main_layout->setContentsMargins(0, 0, 0, 0);

	// PVLAYERSTACKVIEW
	auto* model = new PVLayerStackModel(lib_view);
	auto* delegate = new PVLayerStackDelegate(lib_view, this);
	_layer_stack_view = new PVLayerStackView();
	_layer_stack_view->setItemDelegate(delegate);
	_layer_stack_view->setModel(model);
	_layer_stack_view->resizeColumnsToContents();
	_layer_stack_view->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
	_layer_stack_view->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
	_layer_stack_view->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Interactive);

	// TOOLBAR
	// We create the ToolBar of the PVLayerStackWidget
	layer_stack_toolbar = new QToolBar("Layer Stack ToolBar");
	layer_stack_toolbar->setObjectName("QToolBar_of_PVLayerStackWidget");
	// SIZE STUFF for the ToolBar
	layer_stack_toolbar->setMinimumWidth(185);
	layer_stack_toolbar->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
	// And we fill the ToolBar
	create_actions(layer_stack_toolbar);

	// Now we can add our Widgets to the Layout
	main_layout->addWidget(_layer_stack_view);
	main_layout->addWidget(layer_stack_toolbar);

	setLayout(main_layout);

	/* as layers selectable event count are only needed in the
	 * PVLayerStackWidget, it is a good place to be sure that
	 * existing layers can be processed to compute their
	 * selectable events count.
	 */
	lib_view.recompute_all_selectable_count();
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackWidget::create_actions()
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackWidget::create_actions(QToolBar* toolbar)
{
	QAction* delete_layer_Action;
	QAction* duplicate_layer_Action;
	QAction* move_down_Action;
	QAction* move_up_Action;
	QAction* new_layer_Action;
	QAction* export_all_layers_Action;
	PVLOG_DEBUG("PVGuiQt::PVLayerStackWidget::%s\n", __FUNCTION__);

	// The new_layer Action
	new_layer_Action = new QAction(tr("New Layer"), this);
	new_layer_Action->setIcon(PVModdedIcon("layer-plus"));
	new_layer_Action->setStatusTip(tr("Create a new layer."));
	new_layer_Action->setWhatsThis(tr("Use this to create a new layer."));
	toolbar->addAction(new_layer_Action);
	connect(new_layer_Action, &QAction::triggered, this, &PVLayerStackWidget::new_layer);

	// The duplicate_layer Action
	duplicate_layer_Action = new QAction(tr("Duplicate layer"), this);
	duplicate_layer_Action->setIcon(PVModdedIcon("copy"));
	duplicate_layer_Action->setStatusTip(tr("Duplicate selected layer."));
	duplicate_layer_Action->setToolTip(tr("Duplicate selected layer."));
	duplicate_layer_Action->setWhatsThis(tr("Use this to duplicate the selected layer."));
	toolbar->addAction(duplicate_layer_Action);
	connect(duplicate_layer_Action, &QAction::triggered, this,
	        &PVLayerStackWidget::duplicate_layer);

	// The delete_layer Action
	delete_layer_Action = new QAction(tr("Delete layer"), this);
	delete_layer_Action->setIcon(PVModdedIcon("trash-xmark"));
	delete_layer_Action->setStatusTip(tr("Delete layer."));
	delete_layer_Action->setToolTip(tr("Delete layer."));
	delete_layer_Action->setWhatsThis(tr("Use this to delete the selected."));
	toolbar->addAction(delete_layer_Action);
	connect(delete_layer_Action, &QAction::triggered, this, &PVLayerStackWidget::delete_layer);

	// The move_up Action
	move_up_Action = new QAction(tr("Move up"), this);
	move_up_Action->setIcon(PVModdedIcon("arrow-up-long"));
	move_up_Action->setStatusTip(tr("Move selected layer up."));
	move_up_Action->setToolTip(tr("Move selected layer up."));
	move_up_Action->setWhatsThis(tr("Use this to move the selected layer up."));
	toolbar->addAction(move_up_Action);
	connect(move_up_Action, &QAction::triggered, this, &PVLayerStackWidget::move_up);

	// The move_down Action
	move_down_Action = new QAction(tr("Move down"), this);
	move_down_Action->setIcon(PVModdedIcon("arrow-down-long"));
	move_down_Action->setStatusTip(tr("Move selected layer down."));
	move_down_Action->setToolTip(tr("Move selected layer down."));
	move_down_Action->setWhatsThis(tr("Use this to move the selected layer down."));
	toolbar->addAction(move_down_Action);
	connect(move_down_Action, &QAction::triggered, this, &PVLayerStackWidget::move_down);

	// Export all layers
	const QString& export_all_layers_text = "Export all layers";
	export_all_layers_Action = new QAction(export_all_layers_text, this);
	export_all_layers_Action->setIcon(PVModdedIcon("file-export"));
	export_all_layers_Action->setStatusTip(export_all_layers_text);
	export_all_layers_Action->setToolTip(export_all_layers_text);
	export_all_layers_Action->setWhatsThis(tr("Export each layer as a CSV file"));
	toolbar->addAction(export_all_layers_Action);
	connect(export_all_layers_Action, &QAction::triggered, this, &PVLayerStackWidget::export_all_layers);
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackWidget::export_all_layers
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackWidget::export_all_layers()
{
	PVExportSelectionDlg::export_layers(ls_model()->lib_view());
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackWidget::export_all_layers
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackWidget::delete_layer()
{
	ls_model()->delete_selected_layer();
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackWidget::duplicate_layer
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackWidget::duplicate_layer()
{
	bool& should_hide_layers = ls_model()->lib_layer_stack().should_hide_layers();
	QString name = PVWidgets::PVNewLayerDialog::get_new_layer_name_from_dialog(
	    ls_model()->lib_layer_stack().get_new_layer_name(), should_hide_layers, this);

	if (!name.isEmpty()) {

		if (should_hide_layers) {
			ls_model()->lib_layer_stack().hide_layers();
		}

		ls_model()->duplicate_selected_layer(name);
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackWidget::move_down
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackWidget::move_down()
{
	ls_model()->move_selected_layer_down();
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackWidget::move_up
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackWidget::move_up()
{
	ls_model()->move_selected_layer_up();
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackWidget::new_layer
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackWidget::new_layer()
{
	bool& should_hide_layers = ls_model()->lib_layer_stack().should_hide_layers();
	QString name = PVWidgets::PVNewLayerDialog::get_new_layer_name_from_dialog(
	    ls_model()->lib_layer_stack().get_new_layer_name(), should_hide_layers, this);

	if (!name.isEmpty()) {
		if (should_hide_layers) {
			ls_model()->lib_layer_stack().hide_layers();
		}

		ls_model()->add_new_layer(name);
	}
}

PVGuiQt::PVLayerStackModel* PVGuiQt::PVLayerStackWidget::ls_model()
{
	return _layer_stack_view->ls_model();
}
