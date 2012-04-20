#include <QLabel>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QHBoxLayout>

#include <pvkernel/widgets/PVArgumentListWidget.h>
#include <picviz/widgets/PVAD2GFunctionPropertiesWidget.h>
#include <picviz/widgets/PVArgumentListWidgetFactory.h>
#include <pvkernel/widgets/PVArgumentListWidgetFactory.h>

static QVBoxLayout* layout_from_widget(QWidget* w)
{
	QVBoxLayout *ret = new QVBoxLayout();
	ret->addWidget(w);
	return ret;
}

PVWidgets::PVAD2GFunctionPropertiesWidget::PVAD2GFunctionPropertiesWidget(Picviz::PVView const& view_org, Picviz::PVView const& view_dst, Picviz::PVSelRowFilteringFunction const& rff, QWidget* parent /*= 0*/) :
	QWidget(parent),
	_view_org(view_org),
	_view_dst(view_dst)
{
	// Widgets
	QLabel* function_label = new QLabel(tr("Function: "));
	_function_combo = new QComboBox();

	_global_view_box = new QGroupBox(tr("Global properties"));
	_args_global_widget = new PVWidgets::PVArgumentListWidget(PVWidgets::PVArgumentListWidgetFactory::create_core_widgets_factory());
	_global_view_box->setLayout(layout_from_widget(_args_global_widget));

	_src_view_box = new QGroupBox(tr("Properties for original view"));
	_args_org_widget = new PVWidgets::PVArgumentListWidget(PVWidgets::PVArgumentListWidgetFactory::create_layer_widget_factory(view_org));
	_src_view_box->setLayout(layout_from_widget(_args_org_widget));

	_dst_view_box = new QGroupBox(tr("Properties for destination view"));
	_args_dst_widget = new PVWidgets::PVArgumentListWidget(PVWidgets::PVArgumentListWidgetFactory::create_layer_widget_factory(view_dst));
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

	setLayout(main_layout);

	init_combo_list_rffs(&rff);
	set_current_rff(&rff);

	// Connections
	connect(_function_combo, SIGNAL(currentIndexChanged(int)), this, SLOT(combo_func_changed(int)));
}

void PVWidgets::PVAD2GFunctionPropertiesWidget::combo_func_changed(int idx)
{
	QString plugin = _function_combo->itemData(idx).toString();
	Picviz::PVSelRowFilteringFunction_p lib_rff = LIB_CLASS(Picviz::PVSelRowFilteringFunction)::get().get_class_by_name(plugin);
	if (!lib_rff) {
		return;
	}

	set_current_rff(lib_rff.get());
}

void PVWidgets::PVAD2GFunctionPropertiesWidget::init_combo_list_rffs(Picviz::PVSelRowFilteringFunction const* rff)
{
	LIB_CLASS(Picviz::PVSelRowFilteringFunction)::list_classes const& rffs = LIB_CLASS(Picviz::PVSelRowFilteringFunction)::get().get_list();
	LIB_CLASS(Picviz::PVSelRowFilteringFunction)::list_classes::const_iterator it;
	QString human_name = rff->get_human_name();
	for (it = rffs.begin(); it != rffs.end(); it++) {
		_function_combo->addItem(it.value()->get_human_name(), QVariant(it.key()));
		if (it.value()->get_human_name() == human_name) {
			_function_combo->setCurrentIndex(_function_combo->count()-1);
		}
	}
}

void PVWidgets::PVAD2GFunctionPropertiesWidget::set_current_rff(Picviz::PVSelRowFilteringFunction const* rff)
{
	if (_cur_rff) {
		_rffs_args[*_cur_rff] = _cur_rff->get_args();
	}
	_cur_rff = rff->clone<Picviz::PVSelRowFilteringFunction>();
	map_rff_args_t::const_iterator it = _rffs_args.find(*_cur_rff);
	if (it != _rffs_args.end()) {
		_cur_rff->set_args(it->second);
	}

	_args_global = _cur_rff->get_global_args();
	_args_global_widget->set_args(_args_global);
	_global_view_box->setVisible(_args_global.count() > 0);

	_args_org = _cur_rff->get_args_for_org_view();
	_args_org_widget->set_args(_args_org);
	_src_view_box->setVisible(_args_org.count() > 0);

	_args_dst = _cur_rff->get_args_for_dst_view();
	_args_dst_widget->set_args(_args_dst);
	_dst_view_box->setVisible(_args_dst.count() > 0);
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
}
