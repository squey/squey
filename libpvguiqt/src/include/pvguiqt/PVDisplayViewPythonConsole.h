/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2020
 */

#ifndef PVDISPLAYS_PVDISPLAYVIEWPYTHONCONSOLE_H
#define PVDISPLAYS_PVDISPLAYVIEWPYTHONCONSOLE_H

#include <pvkernel/core/PVRegistrableClass.h>
#include <pvdisplays/PVDisplayIf.h>

namespace PVDisplays
{

class PVDisplayViewPythonConsole : public PVDisplayViewIf
{
  public:
	PVDisplayViewPythonConsole();

  public:
	QWidget*
	create_widget(Inendi::PVView* view, QWidget* parent, Params const& data = {}) const override;

	CLASS_REGISTRABLE(PVDisplayViewPythonConsole)
};
} // namespace PVDisplays

#endif
