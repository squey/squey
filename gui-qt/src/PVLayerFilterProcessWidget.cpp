#include <QDialog>
#include <QLabel>
#include <QPushButton>
#include <QMessageBox>

#include <picviz/PVStateMachine.h>

#include <PVMainWindow.h>
#include <PVLayerFilterProcessWidget.h>
#include <pvkernel/core/PVProgressBox.h>

PVInspector::PVLayerFilterProcessWidget::PVLayerFilterProcessWidget(PVTabSplitter* tab, PVCore::PVArgumentList& args, Picviz::PVLayerFilter_p filter_p) :
	PVArgumentListWidget(*tab->get_lib_view(), args, tab),
	_tab(tab),
	_view(tab->get_lib_view()),
	_filter_p(filter_p),
	_help_btn(NULL),
	_pre_filter_layer_org(_view->pre_filter_layer),
	_args_org(_args),
	_has_apply(false)
{
	setWindowTitle("Filter properties...");
}

PVInspector::PVLayerFilterProcessWidget::~PVLayerFilterProcessWidget()
{
}

void PVInspector::PVLayerFilterProcessWidget::create_btns()
{
	_apply_btn = new QPushButton(QIcon(":/save"),"Apply");
	_preview_btn = new QPushButton(QIcon(":/filter"),"Preview");
	_preview_btn->setDefault(true);
	_cancel_btn = new QPushButton(QIcon(":/red-cross"),"Cancel");
	
	QString filter_desc = _filter_p->detailed_description();
	if (!filter_desc.isEmpty()) {
		_help_btn = new QPushButton(QIcon(":/help"), "Help");	
	}
}

void PVInspector::PVLayerFilterProcessWidget::set_btns_layout()
{
	if (_help_btn) {
		_btn_layout->addWidget(_help_btn);
	}

	_btn_layout->addWidget(_cancel_btn);
	_btn_layout->addWidget(_preview_btn);
	_btn_layout->addWidget(_apply_btn);
}

void PVInspector::PVLayerFilterProcessWidget::connect_btns()
{
	connect(_apply_btn, SIGNAL(pressed()), this, SLOT(save_Slot()));
	connect(_preview_btn, SIGNAL(pressed()), this, SLOT(preview_Slot()));
	connect(_cancel_btn, SIGNAL(pressed()), this, SLOT(cancel_Slot()));
	if (_help_btn) {
		QMessageBox *msgBox = new QMessageBox(QMessageBox::Information, "Filter help", _filter_p->detailed_description(), QMessageBox::Ok, this);
		connect(_help_btn, SIGNAL(pressed()), msgBox, SLOT(exec()));
	}
}

void PVInspector::PVLayerFilterProcessWidget::save_Slot()
{	

	if (_has_apply) {
		// We test if we haven't made a selection different from the one we previewed
		if ((_view->state_machine->get_square_area_mode() != Picviz::PVStateMachine::AREA_MODE_OFF) || args_changed()) {
			if (!process()) {
				// It has been canceled, so don't close the window !
				return;
			}
		}		
	} else {
		if (!process()) {
			return;
		}
	}

	// Save in current layer
	Picviz::PVLayer &current_selected_layer = _view->layer_stack.get_selected_layer();
	/* We fill it's lines_properties */
	// _view->post_filter_layer.A2B_copy_restricted_by_selection_and_nelts(current_selected_layer, _view->real_output_selection, _view->row_count);

	// we change current layer's lines properties with post filter layer's lines properties
	_view->output_layer.get_lines_properties().A2B_copy_restricted_by_selection_and_nelts(current_selected_layer.get_lines_properties(), _view->real_output_selection, _view->row_count);
	// we just process the layer stack
	_view->process_layer_stack();
	// we deactivate the square area
	_view->state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_OFF);
	// we apply the selection
	_view->process_selection();
	// we use the selection of post filter layer
	_view->pre_filter_layer.get_selection() = _view->post_filter_layer.get_selection();
	_view->process_from_filter();

	// We select lines
	// _view->output_layer.get_lines_properties().A2B_copy_restricted_by_selection_and_nelts(current_selected_layer.get_lines_properties(), _view->real_output_selection, _view->row_count);
	/* We need to process the view from the layer_stack */
	// _view->process_from_layer_stack();
	/* We refresh the PVView_p */
	_tab->get_main_window()->update_pvglview(_view, PVGL_COM_REFRESH_SELECTION|PVGL_COM_REFRESH_COLOR);
	_tab->refresh_listing_Slot();

	// FIXME: I think this refreshes the listing too. We shall remove the refresh listing slot then
	_tab->get_main_window()->refresh_view(_view);

	accept();
}

bool PVInspector::PVLayerFilterProcessWidget::process()
{
	_view->process_selection();

	Picviz::PVLayerFilter_p filter_p = _filter_p->clone<Picviz::PVLayerFilter>();
	filter_p->set_args(_args);
	filter_p->set_view(_view);
	filter_p->set_output(&_view->post_filter_layer);

	_view->pre_filter_layer.get_selection() &= _view->layer_stack.get_selected_layer().get_selection();

	PVCore::PVProgressBox *progressDialog = new PVCore::PVProgressBox(tr("Previewing filter..."), this, 0);
	QFuture<void> worker = QtConcurrent::run<void>(process_layer_filter, filter_p.get(), &_view->pre_filter_layer);
	QFutureWatcher<void> watcher;
	watcher.setFuture(worker);
	QObject::connect(&watcher, SIGNAL(finished()), progressDialog, SLOT(accept()), Qt::QueuedConnection);
	
	if(!progressDialog->exec()) {
		// If it has been canceled...
		PVLOG_DEBUG("Filtering action canceled\n");
		disconnect(&watcher, SIGNAL(finished()), 0, 0);

		// Tell the filter that it should stop its processing
		filter_p->cancel();

		// Wait for the filter to finish
		watcher.waitForFinished();

		_view->post_filter_layer = _view->pre_filter_layer;
		return false;
	}

	// We made it ! :)
	PVLOG_DEBUG("Filtering action performed\n");
	// _view->pre_filter_layer = _view->post_filter_layer;
	// _view->state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE);
	_view->state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_OFF);

	// We reprocess the pipeline from the eventline stage
	_view->process_from_eventline();
	_tab->get_main_window()->update_pvglview(_view, PVGL_COM_REFRESH_SELECTION|PVGL_COM_REFRESH_COLOR);
	_tab->refresh_listing_Slot();
	_has_apply = true;
	clear_args_state();

	return true;
}

void PVInspector::PVLayerFilterProcessWidget::preview_Slot()
{
	process();
}

void PVInspector::PVLayerFilterProcessWidget::cancel_Slot()
{
	if (!_has_apply) {
		reject();
		return;
	}

	// Restore original arguments of this layer filter
	_args = _args_org;

	// Restore the original post_filter_layer
	_view->post_filter_layer = _view->pre_filter_layer;

	// Update verything
	_view->process_from_layer_stack();
	_tab->get_main_window()->update_pvglview(_view, PVGL_COM_REFRESH_SELECTION|PVGL_COM_REFRESH_COLOR);
	_tab->refresh_listing_Slot();

	reject();
}

void  PVInspector::PVLayerFilterProcessWidget::process_layer_filter(Picviz::PVLayerFilter* filter, Picviz::PVLayer* layer)
{
	filter->operator()(*layer);
}

