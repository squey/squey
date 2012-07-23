#include "axes-comb_model.h"

void PVViewObserver::refresh()
{
	PVLOG_INFO("PVAxisCombinationObserver::refresh\n");
}

void remove_column_Observer::about_to_be_updated(const arguments_type& args) const
{
	int axis_index = args.get_arg<0>();

	PVLOG_INFO("remove_column_Observer::about_to_be_updated %d\n", axis_index);

	_model->beginRemoveRow(axis_index);
}

void remove_column_Observer::update(const arguments_type& args) const
{
	PVLOG_INFO("remove_column_Observer::update\n");

	_model->endRemoveRow();
}

void axis_append_Observer::about_to_be_updated(const arguments_type& args) const
{
	int axis_index = ((Picviz::PVView*)get_object())->get_axes_count();
	PVLOG_INFO("axis_append_Observer::about_to_be_updated %d\n", axis_index);

	_model->beginInsertRow(axis_index);
}

void axis_append_Observer::update(const arguments_type& args) const
{
	PVLOG_INFO("axis_append_Observer::update\n");

	_model->endInsertRow();
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

	PVLOG_INFO("move_axis_to_new_position_Observer::update %d <-> %d \n", old_index, new_index);

	_model->removeRows(old_index, 1);
	_model->insertRows(new_index, 1);
}
