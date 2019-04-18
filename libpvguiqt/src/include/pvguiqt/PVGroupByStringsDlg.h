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
	                    const create_model_f& f,
	                    const Inendi::PVSelection& sel,
	                    bool counts_are_integers,
	                    QWidget* parent = nullptr)
	    : PVAbstractListStatsDlg(view, c1, f, counts_are_integers, parent)
	    , _col2(c2)
	    , _col2_name(view.get_parent<Inendi::PVSource>().get_format().get_axes().at(c2).get_name())
	    , _sel(sel)
	{
		_ctxt_menu->addSeparator();
		_act_details = new QAction("Show details", _ctxt_menu);
		_ctxt_menu->addAction(_act_details);
	}

	bool process_context_menu(QAction* act) override;

	PVStatsModel*
	details_create_model(const Inendi::PVView& view, PVCol c, Inendi::PVSelection const& sel);

  private:
	PVCol _col2;
	QString _col2_name;
	Inendi::PVSelection _sel; //!< Store selection to be able to compute 'details'
	QAction* _act_details;    //!< Action to show details
};
} // namespace PVGuiQt

#endif // __PVGUIQT_PVCOUNTBYSTRINGSDLG_H__
