/**
 * \file PVAD2GFunctionPropertiesWidget.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <QLabel>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QHBoxLayout>

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/widgets/PVArgumentListWidget.h>
#include <pvkernel/widgets/PVArgumentListWidgetFactory.h>

#include <picviz/PVSelRowFilteringFunction.h>
#include <picviz/widgets/PVAD2GFunctionPropertiesWidget.h>
#include <picviz/widgets/PVArgumentListWidgetFactory.h>

static QVBoxLayout* layout_from_widget(QWidget* w)
{
	QVBoxLayout *ret = new QVBoxLayout();
	ret->addWidget(w);
	return ret;
}

PVWidgets::PVAD2GFunctionPropertiesWidget::PVAD2GFunctionPropertiesWidget(QWidget* parent):
	QWidget(parent),
	_view_org(NULL),
	_view_dst(NULL),
	_args_org_widget(NULL),
	_args_dst_widget(NULL),
	_args_global_widget(NULL)
{
	// Widgets
	QLabel* function_label = new QLabel(tr("Function: "));
	_function_combo = new QComboBox();

	_global_view_box = new QGroupBox(tr("Global properties"));
	_args_global_widget = new PVWidgets::PVArgumentListWidget(PVWidgets::PVArgumentListWidgetFactory::create_core_widgets_factory());
	_global_view_box->setLayout(layout_from_widget(_args_global_widget));

	_src_view_box = new QGroupBox(tr("Properties for original view"));
	_args_org_widget = new PVWidgets::PVArgumentListWidget();
	_src_view_box->setLayout(layout_from_widget(_args_org_widget));

	_dst_view_box = new QGroupBox(tr("Properties for destination view"));
	_args_dst_widget = new PVWidgets::PVArgumentListWidget();
	_dst_view_box->setLayout(layout_from_widget(_args_dst_widget));

	// Layout
	QVBoxLayout* main_layout = new QVBoxLayout();
	QHBoxLayout* function_type_layout = new QHBoxLayout();
	function_type_layout->addWidget(function_label);
	function_type_layout->addWidget(_function_combo);
	main_layout->addLayout(function_type_layout);
	main_layout->addWidget(_global_view_box);
	main_layout->addWidget(_src_view_box);
	main_layout->addWidget(_dst_view_box);

	/*
	_widget = new QWidget();
	_widget->setLayout(main_layout);
	_stacked_widget = new QStackedWidget();

	QVBoxLayout* stacked_widget_layout = new QVBoxLayout();
	stacked_widget_layout->addWidget(_stacked_widget);
	setLayout(stacked_widget_layout);*/
	setLayout(main_layout);

	init_combo_list_rffs();
	_function_combo->setEnabled(false);

	// Connections
	connect(_args_global_widget, SIGNAL(args_changed_Signal()), this, SLOT(commit_args()));
	connect(_args_org_widget,  SIGNAL(args_changed_Signal()), this, SLOT(commit_args()));
	connect(_args_dst_widget,  SIGNAL(args_changed_Signal()), this, SLOT(commit_args()));
	connect(_function_combo, SIGNAL(currentIndexChanged(int)), this, SLOT(combo_func_changed(int)));
}

void PVWidgets::PVAD2GFunctionPropertiesWidget::combo_func_changed(int idx)
{
	if (!_view_org || !_view_dst) {
		return;
	}
	QString plugin = _function_combo->itemData(idx).toString();
	Picviz::PVSelRowFilteringFunction_p lib_rff = LIB_CLASS(Picviz::PVSelRowFilteringFunction)::get().get_class_by_name(plugin);
	if (!lib_rff) {
		return;
	}

	set_current_rff(lib_rff.get());
}

void PVWidgets::PVAD2GFunctionPropertiesWidget::init_combo_list_rffs()
{
	_function_combo->clear();
	LIB_CLASS(Picviz::PVSelRowFilteringFunction)::list_classes const& rffs = LIB_CLASS(Picviz::PVSelRowFilteringFunction)::get().get_list();
	LIB_CLASS(Picviz::PVSelRowFilteringFunction)::list_classes::const_iterator it;
	for (it = rffs.begin(); it != rffs.end(); it++) {
		_function_combo->addItem(it.value()->get_human_name(), QVariant(it.key()));
	}
}

void PVWidgets::PVAD2GFunctionPropertiesWidget::set_args_widget(PVArgumentListWidget* widgets, PVCore::PVArgumentList& args)
{
	widgets->blockSignals(true);
	widgets->set_args(args);
	widgets->blockSignals(false);
}

void PVWidgets::PVAD2GFunctionPropertiesWidget::set_views(Picviz::PVView const& view_org, Picviz::PVView const& view_dst)
{
	_view_org = &view_org;
	_view_dst = &view_dst;

	_args_org_widget->set_widget_factory(PVWidgets::PVArgumentListWidgetFactory::create_layer_widget_factory(view_org));
	_args_dst_widget->set_widget_factory(PVWidgets::PVArgumentListWidgetFactory::create_layer_widget_factory(view_dst));

	set_args_widget(_args_org_widget, _args_org);
	set_args_widget(_args_dst_widget, _args_dst);
}

void PVWidgets::PVAD2GFunctionPropertiesWidget::set_current_rff(Picviz::PVSelRowFilteringFunction const* rff, bool keep_args)
{
	_function_combo->setEnabled(true);

	// Check that the combo box has the good selection
	QString cur_plugin = _function_combo->itemData(_function_combo->currentIndex()).toString();
	if (cur_plugin != rff->registered_name()) {
		int idx = _function_combo->findData(rff->registered_name());
		if (idx >= 0) {
			_function_combo->blockSignals(true);
			_function_combo->setCurrentIndex(idx);
			_function_combo->blockSignals(false);
		}
	}

	if (keep_args) {
		if (_cur_rff) {
			_rffs_args[*_cur_rff] = _cur_rff->get_args();
		}
		_cur_rff = rff->clone<Picviz::PVSelRowFilteringFunction>();
		map_rff_args_t::const_iterator it = _rffs_args.find(*_cur_rff);
		if (it != _rffs_args.end()) {
			_cur_rff->set_args(it->second);
		}
	}
	else {
		_cur_rff = rff->clone<Picviz::PVSelRowFilteringFunction>();
	}

	_args_global = _cur_rff->get_global_args();
	_args_org = _cur_rff->get_args_for_org_view();
	_args_dst = _cur_rff->get_args_for_dst_view();

	set_args_widget(_args_global_widget, _args_global);
	set_args_widget(_args_org_widget, _args_org);
	set_args_widget(_args_dst_widget, _args_dst);

	_global_view_box->setVisible(_args_global.count() > 0);
	_src_view_box->setVisible(_args_org.count() > 0);
	_dst_view_box->setVisible(_args_dst.count() > 0);

	emit function_properties_changed(_cur_rff);
}

void PVWidgets::PVAD2GFunctionPropertiesWidget::commit_args()
{
	PVCore::PVArgumentList args;
	PVCore::PVArgumentList::const_iterator it;
	for (it = _args_global.begin(); it != _args_global.end(); it++) {
		args[it.key()] = it.value();
	}
	for (it = _args_org.begin(); it != _args_org.end(); it++) {
		args[it.key()] = it.value();
	}
	for (it = _args_dst.begin(); it != _args_dst.end(); it++) {
		args[it.key()] = it.value();
	}
	_cur_rff->set_args(args);

	emit function_properties_changed(_cur_rff);
}
