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
