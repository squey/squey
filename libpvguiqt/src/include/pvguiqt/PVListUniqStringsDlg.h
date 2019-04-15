/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2019
 */

#ifndef __PVGUIQT_PVLISTUNIQSTRINGSDLG_H__
#define __PVGUIQT_PVLISTUNIQSTRINGSDLG_H__

#include <inendi/PVSource.h>
#include <inendi/PVView.h>

#include <pvguiqt/PVAbstractListStatsDlg.h>
#include <pvkernel/core/PVProgressBox.h>

#include <pvguiqt/PVStatsModel.h>

#include <pvcop/db/array.h>
#include <pvcop/db/algo.h>

namespace PVGuiQt
{

class PVListUniqStringsDlg : public PVAbstractListStatsDlg
{
  public:
	PVListUniqStringsDlg(Inendi::PVView& view,
	                     PVCol c,
	                     const create_model_f& f,
	                     QWidget* parent = nullptr)
	    : PVAbstractListStatsDlg(view,
	                             c,
	                             f,
	                             true, /* counts_are_integer */
	                             parent)
	{
		QString col1_name =
		    view.get_parent<Inendi::PVSource>().get_format().get_axes().at(c).get_name();
		setWindowTitle("Distinct values of axe '" + col1_name + "'");
	}
};

} // namespace PVGuiQt

#endif // __PVGUIQT_PVLISTUNIQSTRINGSDLG_H__
