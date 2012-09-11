/**
 * \file PVLayerFilterProcessWidget.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVLayerFilterProcessWidget.h"

#include <QDialog>
#include <QLabel>
#include <QPushButton>
#include <QMessageBox>

#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/widgets/PVArgumentListWidget.h>
#include <picviz/PVStateMachine.h>
#include <picviz/widgets/PVArgumentListWidgetFactory.h>

#include <PVMainWindow.h>

PVInspector::PVLayerFilterProcessWidget::PVLayerFilterProcessWidget(PVTabSplitter* tab, PVCore::PVArgumentList& args, Picviz::PVLayerFilter_p filter_p) :
	QDialog(tab),
	_tab(tab),
	_view(tab->get_lib_view()),
	_filter_p(filter_p),
	_presets_widget(NULL),
	_splitter(NULL),
	_help_btn(NULL),
	_pre_filter_layer_org(_view->pre_filter_layer),
	_args_org(args),
	_has_apply(false)
{
	_args_widget = new PVWidgets::PVArgumentListWidget(PVWidgets::PVArgumentListWidgetFactory::create_layer_widget_factory(*tab->get_lib_view()), args, tab);
	setWindowTitle("Filter properties...");
	setObjectName("PVLayerFilterProcessWidget");

	// Presets widget
	if(_filter_p->get_presets().can_have_presets()) {
		_presets_widget = new PVWidgets::PVPresetsWidget(tr("Presets"));
		_presets_widget->add_presets(_filter_p->get_presets().list_presets());
		connect(_presets_widget, SIGNAL(btn_load_clicked_Signal(const QString&)), this, SLOT(load_preset_Slot(const QString&)));
		connect(_presets_widget, SIGNAL(btn_new_clicked_Signal(const QString&)), this, SLOT(add_preset_Slot(const QString&)));
		connect(_presets_widget, SIGNAL(btn_save_clicked_Signal(const QString&)), this, SLOT(save_preset_Slot(const QString&)));
		connect(_presets_widget, SIGNAL(btn_remove_clicked_Signal(const QString&)), this, SLOT(remove_preset_Slot(const QString&)));
		connect(_presets_widget, SIGNAL(preset_renamed_Signal(const QString&, const QString&)), this, SLOT(rename_preset_Slot(const QString&, const QString&)));
		_presets_widget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
		_presets_widget->setMinimumSize(QSize(0, 130));
	}

	// Args widget
	QVBoxLayout* args_widget_box_layout = new QVBoxLayout();
	QGroupBox* args_widget_box = new QGroupBox(tr("Filter"));
	args_widget_box_layout->addWidget(_args_widget);
	args_widget_box->setLayout(args_widget_box_layout);
	_args_widget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);

	// Buttons
	_btn_layout = new QHBoxLayout();
	create_btns();
	set_btns_layout();
	connect_btns();

	// Splitter
	QVBoxLayout* main_layout = new QVBoxLayout();
	if (_filter_p->get_presets().can_have_presets()) {
		_splitter = new QSplitter(Qt::Vertical);
		_splitter->setChildrenCollapsible(false);
		_splitter->addWidget(args_widget_box);
		_splitter->addWidget(_presets_widget);
		_splitter->setStretchFactor(0, 4);
		_splitter->setStretchFactor(1, 1);
		main_layout->addWidget(_splitter);
	}
	else
	{
		main_layout->addWidget(args_widget_box);
		main_layout->addWidget(_presets_widget);
	}

	main_layout->addLayout(_btn_layout);
	setLayout(main_layout);
}

PVInspector::PVLayerFilterProcessWidget::~PVLayerFilterProcessWidget()
{
	_args_widget->deleteLater();
}

void PVInspector::PVLayerFilterProcessWidget::change_args(PVCore::PVArgumentList const& args)
{
	_args_widget->set_args_values(args);
}

void PVInspector::PVLayerFilterProcessWidget::add_preset_Slot(const QString& preset)
{
	_filter_p->set_args(*_args_widget->get_args());
	_filter_p->get_presets().add_preset(preset);
}

void PVInspector::PVLayerFilterProcessWidget::load_preset_Slot(const QString& preset)
{
	_filter_p->set_args(*_args_widget->get_args());
	_filter_p->get_presets().load_preset(preset);
	change_args(_filter_p->get_args());
}

void PVInspector::PVLayerFilterProcessWidget::remove_preset_Slot(const QString& preset)
{
	_filter_p->get_presets().del_preset(preset);
}

void PVInspector::PVLayerFilterProcessWidget::save_preset_Slot(const QString& preset)
{
	_filter_p->set_args(*_args_widget->get_args());
	_filter_p->get_presets().modify_preset(preset);
}

void PVInspector::PVLayerFilterProcessWidget::rename_preset_Slot(const QString& old_preset, const QString& new_preset)
{
	_filter_p->get_presets().rename_preset(old_preset, new_preset);
}

void PVInspector::PVLayerFilterProcessWidget::create_btns()
{
	_apply_btn = new QPushButton(QIcon(":/save"),"Apply");
	_preview_btn = new QPushButton(QIcon(":/filter"),"Preview");
	_preview_btn->setDefault(true);
	_cancel_btn = new QPushButton(QIcon(":/red-cross"),"Cancel");
	_reset_btn = new QPushButton(QIcon(":/document-new"),"Reset");
	_reset_btn->setVisible(_filter_p->get_presets().can_have_presets());
	
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
	_btn_layout->addWidget(_reset_btn);
	_btn_layout->addWidget(_preview_btn);
	_btn_layout->addWidget(_apply_btn);
}

void PVInspector::PVLayerFilterProcessWidget::connect_btns()
{
	connect(_cancel_btn, SIGNAL(pressed()), this, SLOT(cancel_Slot()));
	connect(_reset_btn, SIGNAL(pressed()), this, SLOT(reset_Slot()));
	connect(_preview_btn, SIGNAL(pressed()), this, SLOT(preview_Slot()));
	connect(_apply_btn, SIGNAL(pressed()), this, SLOT(save_Slot()));
	if (_help_btn) {
		QMessageBox *msgBox = new QMessageBox(QMessageBox::Information, "Filter help", _filter_p->detailed_description(), QMessageBox::Ok, this);
		connect(_help_btn, SIGNAL(pressed()), msgBox, SLOT(exec()));
	}
}

void PVInspector::PVLayerFilterProcessWidget::save_Slot()
{
	// Force the current parameter widget to lose its focus (in case it has not been updated yet !)
	_apply_btn->setFocus(Qt::MouseFocusReason);

	if (_has_apply) {
		// We test if we haven't made a selection different from the one we previewed
		if ((_view->state_machine->get_square_area_mode() != Picviz::PVStateMachine::AREA_MODE_OFF) || _args_widget->args_changed()) {
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

	/* We refresh the PVView_p */
	// FIXME: hive

	// FIXME: I think this refreshes the listing too. We shall remove the refresh listing slot then
	//_tab->get_main_window()->refresh_view(_view);

	// Save last used filter
	_view->set_last_used_filter(_filter_p->registered_name());

	accept();
}

bool PVInspector::PVLayerFilterProcessWidget::process()
{
	_view->process_selection();

	Picviz::PVLayerFilter_p filter_p = _filter_p->clone<Picviz::PVLayerFilter>();
	filter_p->set_args(*_args_widget->get_args());
	filter_p->set_view(_view->shared_from_this());
	filter_p->set_output(&_view->post_filter_layer);

	_view->pre_filter_layer.get_selection() &= _view->layer_stack.get_selected_layer().get_selection();

	PVCore::PVProgressBox *progressDialog = new PVCore::PVProgressBox(tr("Previewing filter..."), this, 0);
	QFuture<void> worker = QtConcurrent::run<>(process_layer_filter, filter_p.get(), &_view->pre_filter_layer);
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
	// FIXME: hive !
	//_tab->get_main_window()->//update_pvglview(_view, PVSDK_MESSENGER_REFRESH_SELECTION|PVSDK_MESSENGER_REFRESH_COLOR);
	//_tab->refresh_listing_Slot();
	_has_apply = true;
	_args_widget->clear_args_state();

	return true;
}

void PVInspector::PVLayerFilterProcessWidget::preview_Slot()
{
	// Force the current parameter widget to lose its focus (in case it has not been updated yet !)
	_preview_btn->setFocus(Qt::MouseFocusReason);

	process();
}

void PVInspector::PVLayerFilterProcessWidget::cancel_Slot()
{
	if (!_has_apply) {
		reject();
		return;
	}

	// Restore original arguments of this layer filter
	*_args_widget->get_args() = _args_org;

	// Restore the original post_filter_layer
	_view->post_filter_layer = _view->pre_filter_layer;

	// Update everything
	_view->process_from_layer_stack();
	// FIXME: Hive!
	//_tab->get_main_window()->//update_pvglview(_view, PVSDK_MESSENGER_REFRESH_SELECTION|PVSDK_MESSENGER_REFRESH_COLOR);
	//_tab->refresh_listing_Slot();

	reject();
}

void PVInspector::PVLayerFilterProcessWidget::reset_Slot()
{
	change_args(_filter_p->get_default_args_for_view(*_view));
}

void  PVInspector::PVLayerFilterProcessWidget::process_layer_filter(Picviz::PVLayerFilter* filter, Picviz::PVLayer* layer)
{
	filter->operator()(*layer);
}

