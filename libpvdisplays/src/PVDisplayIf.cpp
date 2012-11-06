#include <pvdisplays/PVDisplayIf.h>


// PVDisplayViewAxisIf implementation
QWidget* PVDisplays::PVDisplayViewAxisIf::get_unique_widget(Picviz::PVView* view, PVCol axis_comb, QWidget* parent)
{
	QWidget* ret;
	map_widgets_t::const_iterator it = _widgets.find(Params(view, axis_comb));

	if (it == _widgets.end()) {
		ret = create_widget(view, axis_comb, parent);
		_widgets[Params(view, axis_comb)] = ret;
	}
	else {
		ret = it->second;
		assert(ret->parent() == parent);
	}

	return ret;
}
