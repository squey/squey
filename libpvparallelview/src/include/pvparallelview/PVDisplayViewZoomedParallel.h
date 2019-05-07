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

class PVDisplayViewZoomedParallel : public PVDisplayViewIf
{
  public:
	PVDisplayViewZoomedParallel();

  public:
	QWidget*
	create_widget(Inendi::PVView* view, QWidget* parent, Params const& data = {}) const override;
	void add_to_axis_menu(QMenu& menu,
	                      PVCol axis,
	                      PVCombCol axis_comb,
	                      Inendi::PVView* view,
	                      PVDisplaysContainer* container) override;

	CLASS_REGISTRABLE(PVDisplayViewZoomedParallel)
};
} // namespace PVDisplays

#endif // PVDISPLAYS_PVDISPLAYVIEWZOOMEDPARALLEL_H
