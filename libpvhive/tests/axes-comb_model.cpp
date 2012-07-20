#include "axes-comb_model.h"

void PVViewObserver::refresh()
{
	PVLOG_INFO("PVAxisCombinationObserver::refresh\n");
}

PVHIVE_CALL_OBJECT_BLOCK_BEGIN()

template <>
void PVHive::PVHive::call_object<Picviz::PVView, decltype(&Picviz::PVView::remove_column), &Picviz::PVView::remove_column>
	(Picviz::PVView* view, PVCol axis)
{
	PVLOG_INFO("PVHive::call_object for PVView::remove_column\n");

	// In this case we need to update the model *before* performing the function call (see QAbstractItemModel::beginRemoveRows doc)
	// TODO: We obviously need a method that calls the object without refreshing its observers!
	refresh_func_observers<Picviz::PVView, decltype(&Picviz::PVView::remove_column), &Picviz::PVView::remove_column>(view,axis);

	call_object_default<Picviz::PVView, decltype(&Picviz::PVView::remove_column), &Picviz::PVView::remove_column>(view, axis);
}

PVHIVE_CALL_OBJECT_BLOCK_END()

void remove_column_Observer::update(const arguments_type& args) const
{
	int axis_index = args.get_arg<0>();

	PVLOG_INFO("remove_column_Observer::update %d\n", axis_index);

	_model->removeRows(axis_index, 1);
}

void axis_append_Observer::update(const arguments_type& args) const
{
	int axis_index = ((Picviz::PVView*)get_object())->get_axes_count();
	PVLOG_INFO("axis_append_Observer::update %d\n", axis_index);

	_model->insertRows(axis_index, 1);
}

void set_axis_name_Observer::update(arguments_type const& args) const
{
	int axis_index = args.get_arg<0>();

	PVLOG_INFO("set_axis_name_Observer::update %d\n", axis_index);

	emit const_cast<AxesCombinationListModel*>(_model)->dataChanged(_model->index(axis_index, 0), _model->index(axis_index, 0));
}

void move_axis_to_new_position_Observer::update(arguments_type const& args) const
{
	int old_index = args.get_arg<0>();
	int new_index = args.get_arg<1>();

	PVLOG_INFO("move_axis_to_new_position_Observer::update %d -> %d \n", old_index, new_index);

	_model->removeRows(old_index, 1);
	_model->insertRows(new_index, 1);
}
