/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVDISPLAYS_PVDISPLAYSCONTAINER_H
#define PVDISPLAYS_PVDISPLAYSCONTAINER_H

#include <pvbase/general.h>
#include <QMainWindow>

namespace Inendi
{
class PVView;
}

namespace PVDisplays
{

class PVDisplayViewIf;
class PVDisplayViewDataIf;
class PVDisplayViewZoneIf;

class PVDisplaysContainer : public QMainWindow
{
	Q_OBJECT

  public:
	explicit PVDisplaysContainer(QWidget* w) : QMainWindow(w) {}

  public Q_SLOTS:
	virtual void create_view_widget(PVDisplays::PVDisplayViewIf& interface,
	                                Inendi::PVView* view) = 0;
	virtual void create_view_axis_widget(PVDisplays::PVDisplayViewDataIf& interface,
	                                     Inendi::PVView* view,
	                                     PVCombCol axis_comb) = 0;
	virtual void create_view_zone_widget(PVDisplays::PVDisplayViewZoneIf& interface,
	                                     Inendi::PVView* view,
	                                     PVCombCol zone_index_first,
	                                     PVCombCol zone_index_second) = 0;
};
} // namespace PVDisplays

#endif
