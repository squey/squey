/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVDISPLAYS_PVDISPLAYSOURCELISTING_H
#define PVDISPLAYS_PVDISPLAYSOURCELISTING_H

#include <pvkernel/core/PVRegistrableClass.h>
#include <pvdisplays/PVDisplayIf.h>

namespace PVDisplays
{

class PVDisplaySourceDataTree : public PVDisplaySourceIf
{
  public:
	PVDisplaySourceDataTree();

  public:
	QWidget*
	create_widget(Inendi::PVSource* src, QWidget* parent, Params const& data = {}) const override;

	CLASS_REGISTRABLE(PVDisplaySourceDataTree)
};
} // namespace PVDisplays

#endif
