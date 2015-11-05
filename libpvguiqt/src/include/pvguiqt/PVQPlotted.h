/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVGUIQT_PVQPLOTTED_H
#define PVGUIQT_PVQPLOTTED_H

#include <pvkernel/core/general.h>

namespace Inendi {
class PVPlotted;
}

namespace PVGuiQt {

struct PVQPlotted
{
	static bool edit_plotted(Inendi::PVPlotted& plotted, QWidget* parent = NULL);
};

}

#endif
