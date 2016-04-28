/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVDISPLAYS_PVDISPLAYVIEWZOOMEDPARALLEL_H
#define PVDISPLAYS_PVDISPLAYVIEWZOOMEDPARALLEL_H

#include <pvkernel/core/PVRegistrableClass.h>
#include <pvdisplays/PVDisplayIf.h>

namespace PVDisplays
{

class PVDisplayViewZoomedParallel : public PVDisplayViewAxisIf
{
  public:
	PVDisplayViewZoomedParallel();

  public:
	QWidget* create_widget(Inendi::PVView* view, PVCol axis_comb, QWidget* parent) const override;
	QIcon toolbar_icon() const override;
	QString widget_title(Inendi::PVView* view, PVCol axis_comb) const override;
	QString axis_menu_name(Inendi::PVView const* view, PVCol axis_comb) const override;

	CLASS_REGISTRABLE(PVDisplayViewZoomedParallel)
};
}

#endif // PVDISPLAYS_PVDISPLAYVIEWZOOMEDPARALLEL_H
