#include "axes-comb_model.h"

// For reference: PVACTOR_CALL(*actor, &Picviz::PVView::set_axis_name, rand() % _view_p->get_axes_count(), QString::number(rand() % 1000));

/*PVHIVE_CALL_OBJECT_BLOCK_BEGIN()

template <>
void PVHive::PVHive::call_object<Picviz::PVView, decltype(&Picviz::PVView::set_axis_name), &Picviz::PVView::set_axis_name>
	(Picviz::PVView* view, PVCol axis, boost::reference_wrapper<QString const> axis_name)
{
	PVLOG_INFO("PVHive::call_object for PVView::set_axis_name\n");

	call_object_default<Picviz::PVView, decltype(&Picviz::PVView::set_axis_name), &Picviz::PVView::set_axis_name>(view, axis, axis_name);
	refresh_observers(&view->get_axis_name(axis));

	//static_assert(false, "test"); Ok this is compiling
}

PVHIVE_CALL_OBJECT_BLOCK_END()*/

void PVViewObserver::refresh()
{
	PVLOG_INFO("PVAxisCombinationObserver::refresh\n");
	_model->reset();
}
