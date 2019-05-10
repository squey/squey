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
#include <vector>
#include <any>

namespace Inendi
{
class PVView;
}

namespace PVDisplays
{

class PVDisplayViewIf;

class PVDisplaysContainer : public QMainWindow
{
	Q_OBJECT

  public:
	explicit PVDisplaysContainer(QWidget* w) : QMainWindow(w) {}

  public Q_SLOTS:
	virtual void create_view_widget(PVDisplays::PVDisplayViewIf& interface,
	                                Inendi::PVView* view,
	                                std::vector<std::any> params = {}) = 0;
};
} // namespace PVDisplays

#endif
