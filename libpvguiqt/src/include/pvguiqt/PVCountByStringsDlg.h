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

class PVCountByStringsDlg;

namespace __impl {

class PVCountByStringsModel;
class PVCountByStringsDetailsDlg;

class PVCountByStringsModel : public PVGuiQt::__impl::PVAbstractListStatsModel
{
	friend class PVGuiQt::PVCountByStringsDlg;
	typedef typename PVRush::PVNraw::count_by_t count_by_t;

public:
	PVCountByStringsModel(count_by_t& values, QWidget* parent = nullptr) : PVGuiQt::__impl::PVAbstractListStatsModel(parent), _values(values) {}

	int rowCount(QModelIndex const& parent) const;
	QVariant data(QModelIndex const& index, int role) const;

private:


private:
	count_by_t _values;
};

}

class PVCountByStringsDlg : public PVAbstractListStatsDlg
{
	typedef typename PVRush::PVNraw::count_by_t count_by_t;

public:
	PVCountByStringsDlg(Picviz::PVView_sp& view, PVCol col1, PVCol col2, count_by_t& values, size_t abs_max, size_t rel_min, size_t rel_max, QWidget* parent = nullptr) :
		PVAbstractListStatsDlg(view, col1, new __impl::PVCountByStringsModel(values), abs_max, rel_min, rel_max, parent),
		_view(*view), _col2(col2)
	{
		_ctxt_menu->addSeparator();
		_act_list_v2 = new QAction("Show details", _ctxt_menu);
		_ctxt_menu->addAction(_act_list_v2);
	}

	bool process_context_menu(QAction* act);

private:
	__impl::PVCountByStringsModel* get_model();

private:
	Picviz::PVView& _view;
	PVCol _col2;
	QAction* _act_list_v2;
};



}

#endif // __PVGUIQT_PVCOUNTBYSTRINGSDLG_H__
