/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef __STATS_LISTING_H__
#define __STATS_LISTING_H__

#include <QObject>

#include <inendi/PVView.h>
#include <pvhive/PVHive.h>
#include <pvhive/PVCallHelper.h>

#include <iostream>

struct ViewSlots : public QObject
{
	Q_OBJECT

  public:
	ViewSlots(Inendi::PVView& view) : _view(view) {}

  public slots:
	void select_all()
	{
		std::cout << "change_axes_combination" << std::endl;
		_view.select_all_nonzb_lines();
		Inendi::PVView_sp view_sp = _view.shared_from_this();
		PVHive::PVCallHelper::call<FUNC(Inendi::PVView::process_real_output_selection)>(view_sp);
	}

	void change_axes_combination()
	{
		std::cout << "change_axes_combination" << std::endl;
		_view.remove_column(0);
		_view.axis_append(Inendi::PVAxis());
		_view.axis_append(Inendi::PVAxis());

		Inendi::PVAxesCombination const& axes_combination = _view.get_axes_combination();
		auto& axes_index_list = axes_combination.get_axes_index_list();
		auto& axes_list = axes_combination.get_axes_list();
		Inendi::PVView_sp view_sp = _view.shared_from_this();
		PVHive::PVCallHelper::call<FUNC(Inendi::PVView::set_axes_combination_list_id)>(
		    view_sp, axes_index_list, axes_list);
	}

  private:
	Inendi::PVView& _view;
};

#endif // __STATS_LISTING_H__
