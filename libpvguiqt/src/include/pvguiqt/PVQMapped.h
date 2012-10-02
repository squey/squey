#ifndef PVGUIQT_PVMAPPED_H
#define PVGUIQT_PVMAPPED_H

#include <pvkernel/core/general.h>

namespace Picviz {
class PVMapped;
}

namespace PVGuiQt {

struct PVQMapped
{
	static bool edit_mapped(Picviz::PVMapped& plotted, QWidget* parent = NULL);
};

}

#endif
