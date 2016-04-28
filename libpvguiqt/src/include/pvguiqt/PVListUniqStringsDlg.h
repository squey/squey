/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef __PVGUIQT_PVLISTUNIQSTRINGSDLG_H__
#define __PVGUIQT_PVLISTUNIQSTRINGSDLG_H__

#include <pvguiqt/PVAbstractListStatsDlg.h>

#include <pvguiqt/PVStatsModel.h>

#include <inendi/PVView_types.h>

#include <pvcop/db/array.h>

namespace PVGuiQt
{

class PVListUniqStringsDlg : public PVAbstractListStatsDlg
{
  public:
	PVListUniqStringsDlg(Inendi::PVView_sp& view, PVCol c, pvcop::db::array col1,
	                     pvcop::db::array col2, double abs_max, double rel_min, double rel_max,
	                     QWidget* parent = nullptr)
	    : PVAbstractListStatsDlg(
	          view, c,
	          new PVStatsModel(std::move(col1), std::move(col2), abs_max, rel_min, rel_max), parent)
	{
	}
};
}

#endif // __PVGUIQT_PVLISTUNIQSTRINGSDLG_H__
