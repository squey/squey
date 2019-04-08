/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVDISPLAYS_PVDISPLAYVIEWHITCOUNT_H
#define PVDISPLAYS_PVDISPLAYVIEWHITCOUNT_H

#include <pvkernel/core/PVRegistrableClass.h>
#include <pvdisplays/PVDisplayIf.h>

namespace PVDisplays
{

class PVDisplayViewHitCount : public PVDisplayViewDataIf
{
  public:
	PVDisplayViewHitCount();

  public:
	QWidget*
	create_widget(Inendi::PVView* view, Params const& data, QWidget* parent) const override;
	QIcon toolbar_icon() const override;
	QString widget_title(Inendi::PVView* view, Params const& data) const override;
	QString axis_menu_name(Inendi::PVView* view, Params const& data) const override;

	CLASS_REGISTRABLE(PVDisplayViewHitCount)
};
} // namespace PVDisplays

#endif // PVDISPLAYS_PVDISPLAYVIEWHITCOUNT_H
