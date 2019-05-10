/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVDISPLAYS_PVDISPLAYVIEWFULLPARALLEL_H
#define PVDISPLAYS_PVDISPLAYVIEWFULLPARALLEL_H

#include <pvkernel/core/PVRegistrableClass.h>
#include <pvdisplays/PVDisplayIf.h>

namespace PVDisplays
{

class PVDisplayViewFullParallel : public PVDisplayViewIf
{
  public:
	PVDisplayViewFullParallel();

  public:
	QWidget*
	create_widget(Inendi::PVView* view, QWidget* parent, Params const& data = {}) const override;

	CLASS_REGISTRABLE(PVDisplayViewFullParallel)
};
} // namespace PVDisplays

#endif
