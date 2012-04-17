#include <picviz/PVAD2GFunctionPropertiesWidget.h>


Picviz::PVAD2GFunctionPropertiesWidget::PVAD2GFunctionPropertiesWidget(Picviz::PVView const& view_org, Picviz::PVView const& view_dst, PVSelRowFilteringFunction_p rff_p rff, QWidget* parent /*= 0*/) :
	QWidget(parent),
	_view_org(view_org),
	_view_dst(view_dst)
{
	// Widgets
	QLabel* function_label = new QLabel(tr("Function: "));
	QComboBox* function_combo = new QComboBox();
	QGroupBox* src_view_box = new QGroupBox(tr("Properties for original view"));
	QGroupBox* dst_view_box = new QGroupBox(tr("Properties for destination view"));

	// Layout
	QVBoxLayout* main_layout = new QVBoxLayout();
	QHBoxLayout* function_type_layout = new QHBoxLayout();
	function_type_layout->addWidget(function_label);
	function_type_layout->addWidget(function_combo);
	main_layout->addLayout(function_type_layout);
	main_layout->addWidget(src_view_box);
	main_layout->addWidget(dst_view_box);

	setLayout(main_layout);

	set_current_rff(rff);
}

void Picviz::PVAD2GFunctionPropertiesWidget::set_current_rff(PVSelRowFilteringFunction_p const& rff)
{
	if (_cur_rff) {
		_rffs_args[*_cur_rff] = 
	_cur_rff = rff->clone<PVSelRowFilteringFunction>();
	maps_rff_args_t::const_iterator it = _rffs_args.find(*_cur_rff);
	if (it != _rffs_args.end()) {
		_cur_rff->set_args(
}
