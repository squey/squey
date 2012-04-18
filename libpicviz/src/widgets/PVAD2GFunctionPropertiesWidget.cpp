#include <QLabel>
#include <QComboBox>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QHBoxLayout>

#include <pvkernel/widgets/PVArgumentListWidget.h>
#include <picviz/widgets/PVAD2GFunctionPropertiesWidget.h>
#include <picviz/widgets/PVArgumentListWidgetFactory.h>

PVWidgets::PVAD2GFunctionPropertiesWidget::PVAD2GFunctionPropertiesWidget(Picviz::PVView const& view_org, Picviz::PVView const& view_dst, Picviz::PVSelRowFilteringFunction const& rff, QWidget* parent /*= 0*/) :
	QWidget(parent),
	_view_org(view_org),
	_view_dst(view_dst)
{
	// Widgets
	QLabel* function_label = new QLabel(tr("Function: "));
	QComboBox* function_combo = new QComboBox();
	QGroupBox* src_view_box = new QGroupBox(tr("Properties for original view"));
	QGroupBox* dst_view_box = new QGroupBox(tr("Properties for destination view"));
	_args_org_widget = new PVWidgets::PVArgumentListWidget(PVWidgets::PVArgumentListWidgetFactory::create_layer_widget_factory(view_org));
	_args_dst_widget = new PVWidgets::PVArgumentListWidget(PVWidgets::PVArgumentListWidgetFactory::create_layer_widget_factory(view_dst));

	// Layout
	QVBoxLayout* main_layout = new QVBoxLayout();
	QHBoxLayout* function_type_layout = new QHBoxLayout();
	function_type_layout->addWidget(function_label);
	function_type_layout->addWidget(function_combo);
	main_layout->addLayout(function_type_layout);
	main_layout->addWidget(src_view_box);
	main_layout->addWidget(dst_view_box);

	setLayout(main_layout);

	set_current_rff(&rff);
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

	_args_org = _cur_rff->get_args_for_org_view();
	_args_org_widget->set_args_values(_args_org);

	_args_dst = _cur_rff->get_args_for_dst_view();
	_args_dst_widget->set_args_values(_args_dst);
}
