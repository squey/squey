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

class PVDisplayViewHitCount : public PVDisplayViewIf
{
  public:
	PVDisplayViewHitCount();

  public:
	QWidget*
	create_widget(Inendi::PVView* view, QWidget* parent, Params const& data = {}) const override;

	CLASS_REGISTRABLE(PVDisplayViewHitCount)
};
} // namespace PVDisplays

#endif // PVDISPLAYS_PVDISPLAYVIEWHITCOUNT_H
