/**
 * \file stats_listing.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __STATS_LISTING_H__
#define __STATS_LISTING_H__

#include <QObject>

#include <picviz/PVView.h>
#include <pvhive/PVHive.h>
#include <pvhive/PVCallHelper.h>

#include <iostream>

struct ViewSlots : public QObject {
	Q_OBJECT

public:
	ViewSlots(Picviz::PVView& view) : _view(view) {}

public slots:
	void select_all()
	{
		std::cout << "change_axes_combination" << std::endl;
		_view.select_all_nonzb_lines();
		Picviz::PVView_sp view_sp = _view.shared_from_this();
		PVHive::PVCallHelper::call<FUNC(Picviz::PVView::process_real_output_selection)>(view_sp);
	}

	void change_axes_combination()
	{
		std::cout << "change_axes_combination" << std::endl;
		Picviz::PVAxesCombination& axes_combination = _view.get_axes_combination();
		_view.remove_column(0);
		axes_combination.axis_append(1);
		axes_combination.axis_append(2);

		auto& axes_index_list = axes_combination.get_axes_index_list();
		auto& axes_list = axes_combination.get_axes_list();
		Picviz::PVView_sp view_sp = _view.shared_from_this();
		PVHive::PVCallHelper::call<FUNC(Picviz::PVView::set_axes_combination_list_id)>(view_sp, axes_index_list, axes_list);
	}

private:
	Picviz::PVView& _view;
};


#endif // __STATS_LISTING_H__
