/**
 * \file PVCountByStringsDlg.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef __PVGUIQT_PVCOUNTBYSTRINGSDLG_H__
#define __PVGUIQT_PVCOUNTBYSTRINGSDLG_H__

#include <pvguiqt/PVAbstractListStatsDlg.h>
#include <pvguiqt/PVListUniqStringsDlg.h>

#include <QAbstractListModel>
#include <QMenu>

namespace PVGuiQt
{

namespace __impl
{
class PVCountByStringsModel;
class PVCountByStringsDetailsDlg;
}

class PVCountByStringsDlg : public PVAbstractListStatsDlg
{
public:
	template <typename T>
	PVCountByStringsDlg(Picviz::PVView_sp& view, PVCol col1, PVCol col2, T& values, size_t v2_unique_values_count, QWidget* parent = nullptr) :
		PVAbstractListStatsDlg(view, col1, new __impl::PVCountByStringsModel(values), v2_unique_values_count, parent),
		_view(*view), _col2(col2)
	{
		init_max_element();

		_ctxt_menu->addSeparator();
		_act_list_v2 = new QAction("Show details", _ctxt_menu);
		_ctxt_menu->addAction(_act_list_v2);
	}

	void process_context_menu(QAction* act);

private:
	__impl::PVCountByStringsModel* get_model();
	void init_max_element();

private:
	Picviz::PVView& _view;
	PVCol _col2;
	QAction* _act_list_v2;
};

namespace __impl {

class PVCountByStringsModel : public PVGuiQt::__impl::PVAbstractListStatsModel
{
	friend class PVGuiQt::PVCountByStringsDlg;

public:
	template <typename T>
	PVCountByStringsModel(T& values, QWidget* parent = nullptr) : PVGuiQt::__impl::PVAbstractListStatsModel(parent)
	{
		size_t v1_count = values.size();
		_values.reserve(v1_count);
		for (auto& v1 : values) {
			auto& v2_values = v1.second;

			// v1 count
			string_count_t v1_count_pair(std::move(v1.first) , v2_values.size());

			_values.emplace_back(std::move(v1_count_pair), std::move(v2_values));
		}
	}

	int rowCount(QModelIndex const& parent) const;
	QVariant data(QModelIndex const& index, int role) const;

private:
	size_t get_max_element();

private:
	typedef std::pair<std::string_tbb, size_t> string_count_t;
	typedef std::vector<string_count_t> vector_v2_count_t;
	typedef std::pair<string_count_t, PVRush::PVNraw::unique_values_t > v1_v2_pair_t;
	typedef std::vector<v1_v2_pair_t> count_by_t;

	count_by_t _values;
};

}

}

#endif // __PVGUIQT_PVCOUNTBYSTRINGSDLG_H__
