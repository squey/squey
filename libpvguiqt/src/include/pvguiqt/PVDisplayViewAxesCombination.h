/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVDISPLAYS_PVDISPLAYVIEWAXESCOMBINATION_H
#define PVDISPLAYS_PVDISPLAYVIEWAXESCOMBINATION_H

#include <pvkernel/core/PVRegistrableClass.h>
#include <pvdisplays/PVDisplayIf.h>

namespace PVDisplays
{

class PVDisplayViewAxesCombination : public PVDisplayViewIf
{
  public:
	PVDisplayViewAxesCombination();

  public:
	QWidget* create_widget(Inendi::PVView* view, QWidget* parent) const override;

	CLASS_REGISTRABLE(PVDisplayViewAxesCombination)
};
} // namespace PVDisplays

#endif
