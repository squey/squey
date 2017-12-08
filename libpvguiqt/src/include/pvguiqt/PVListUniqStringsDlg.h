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

#include <pvcop/db/array.h>

namespace PVGuiQt
{

class PVListUniqStringsDlg : public PVAbstractListStatsDlg
{
  public:
	PVListUniqStringsDlg(Inendi::PVView& view,
	                     const QString& col1_name,
	                     PVCol c,
	                     pvcop::db::array col1,
	                     pvcop::db::array col2,
	                     pvcop::db::array abs_max,
	                     pvcop::db::array minmax,
	                     QWidget* parent = nullptr)
	    : PVAbstractListStatsDlg(view,
	                             c,
	                             new PVStatsModel("Count",
	                                              col1_name,
	                                              QString(),
	                                              std::move(col1),
	                                              std::move(col2),
	                                              std::move(abs_max),
	                                              std::move(minmax)),
	                             parent)
	{
	}
};
} // namespace PVGuiQt

#endif // __PVGUIQT_PVLISTUNIQSTRINGSDLG_H__
