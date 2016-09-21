/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef __PVGUIQT_PVCOUNTBYSTRINGSDLG_H__
#define __PVGUIQT_PVCOUNTBYSTRINGSDLG_H__

#include <pvguiqt/PVAbstractListStatsDlg.h>
#include <pvguiqt/PVListUniqStringsDlg.h>

#include <QAbstractListModel>
#include <QMenu>

namespace PVGuiQt
{

class PVGroupByStringsDlg : public PVAbstractListStatsDlg
{
  public:
	PVGroupByStringsDlg(Inendi::PVView& view,
	                    PVCol c1,
	                    PVCol c2,
	                    const Inendi::PVSelection& sel,
	                    pvcop::db::array col1,
	                    pvcop::db::array col2,
	                    double abs_max,
	                    double rel_min,
	                    double rel_max,
	                    QWidget* parent = nullptr)
	    : PVAbstractListStatsDlg(
	          view,
	          c1,
	          new PVStatsModel(std::move(col1), std::move(col2), abs_max, rel_min, rel_max),
	          parent)
	    , _col2(c2)
	    , _sel(sel)
	{
		_ctxt_menu->addSeparator();
		_act_details = new QAction("Show details", _ctxt_menu);
		_ctxt_menu->addAction(_act_details);
	}

	bool process_context_menu(QAction* act) override;

  private:
	PVCol _col2;
	Inendi::PVSelection _sel; //!< Store selection to be able to compute 'details'
	QAction* _act_details;    //!< Action to show details
};
} // namespace PVGuiQt

#endif // __PVGUIQT_PVCOUNTBYSTRINGSDLG_H__
