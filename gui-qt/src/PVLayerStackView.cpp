//! \file PVLayerStackView.cpp
//! $Id: PVLayerStackView.cpp 2501 2011-04-25 14:56:00Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QtGui>
#include <QEvent>

#include <pvkernel/core/general.h>
//#include <picviz/PVView.h>

#include <PVMainWindow.h>

#include <PVLayerStackView.h>

/******************************************************************************
 *
 * PVInspector::PVLayerStackView::PVLayerStackView
 *
 *****************************************************************************/
PVInspector::PVLayerStackView::PVLayerStackView(PVMainWindow *mw, PVLayerStackModel *model, PVLayerStackWidget *parent) : QTableView(parent)
{
	PVLOG_DEBUG("PVInspector::PVLayerStackView::%s\n", __FUNCTION__);

	main_window = mw;

	setMinimumSize(0,0);
	setSizePolicy(QSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding));
	setFocusPolicy(Qt::NoFocus);
	verticalHeader()->hide();
	horizontalHeader()->hide();
	//viewport()->setMouseTracking(true);
	//viewport()->setAttribute(Qt::WA_Hover, true);

	layer_stack_delegate = new PVLayerStackDelegate(mw, this);
	setItemDelegate(layer_stack_delegate);

	layer_stack_event_filter = new PVLayerStackEventFilter(mw, this);
	viewport()->installEventFilter(layer_stack_event_filter);

	mouse_hover_layer_index = -1;
	last_mouse_hover_layer_index = -1;

	// Context menu
	connect(this, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(show_ctxt_menu(const QPoint&)));
	setContextMenuPolicy(Qt::CustomContextMenu);

	_ctxt_menu = new QMenu(this);
	_ctxt_menu_save_act = new QAction(tr("Save this layer..."), NULL);
	_ctxt_menu_load_act = new QAction(tr("Import a layer..."), NULL);
	_ctxt_menu->addAction(_ctxt_menu_save_act);
	_ctxt_menu->addAction(_ctxt_menu_load_act);

	setModel(model);
	resizeColumnsToContents();
	//resizeRowsToContents();
}

/******************************************************************************
 *
 * PVInspector::PVLayerStackView::leaveEvent
 *
 *****************************************************************************/
void PVInspector::PVLayerStackView::leaveEvent(QEvent * /*event*/)
{
	PVLOG_DEBUG("PVInspector::PVLayerStackView::%s\n", __FUNCTION__);

	mouse_hover_layer_index = -1;
	last_mouse_hover_layer_index = -1;
	viewport()->update();
}

void PVInspector::PVLayerStackView::show_ctxt_menu(const QPoint& pt)
{
	QModelIndex idx_click = indexAt(pt);
	_ctxt_menu_save_act->setEnabled(idx_click.isValid());

	QAction* act = _ctxt_menu->exec(QCursor::pos());
	if (act == _ctxt_menu_save_act) {
		save_layer(idx_click.row());
	}
	else
	if (act == _ctxt_menu_load_act) {
		import_layer();
	}
}

void PVInspector::PVLayerStackView::save_layer(int /*idx*/)
{
	// Get layer with index 'idx'
	Picviz::PVLayer& layer = ((PVLayerStackModel*) model())->get_layer_stack_lib().get_selected_layer();

	// Ask for a filename
	QString file = QFileDialog::getSaveFileName(this, tr("Choose a file..."), QString(), "Picviz layer files (*." PICVIZ_LAYER_ARCHIVE_EXT ");;All files (*.*)");
	if (file.isEmpty()) {
		return;
	}

	layer.save_to_file(file);
}

void PVInspector::PVLayerStackView::import_layer()
{
	QString file = QFileDialog::getOpenFileName(this, tr("Import a layer..."), QString(), "Picviz layer files (*." PICVIZ_LAYER_ARCHIVE_EXT ");;All files (*.*)");
	if (file.isEmpty()) {
		return;
	}

	// Create a new layer
	PVLayerStackModel* model_ = (PVLayerStackModel*) model();
	Picviz::PVLayer* layer = model_->get_layer_stack_lib().append_new_layer();

	// And load it
	layer->load_from_file(file);

	model_->emit_layoutChanged();
}
