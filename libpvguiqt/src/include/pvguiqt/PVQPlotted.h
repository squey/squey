/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVGUIQT_PVQPLOTTED_H
#define PVGUIQT_PVQPLOTTED_H

#include <pvkernel/core/general.h>

namespace Picviz {
class PVPlotted;
}

namespace PVGuiQt {

struct PVQPlotted
{
	static bool edit_plotted(Picviz::PVPlotted& plotted, QWidget* parent = NULL);
};

}

#endif
