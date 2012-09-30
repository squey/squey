#ifndef PVGUIQT_PVQNRAW_H
#define PVGUIQT_PVQNRAW_H

#include <pvkernel/core/general.h>

namespace PVRush {
class PVNraw;
}

namespace Picviz {
class PVSelection;
}

namespace PVGuiQt {

class PVListUniqStringsDlg;

struct PVQNraw
{
	static PVListUniqStringsDlg* show_unique_values(PVRush::PVNraw const& nraw, PVCol c, Picviz::PVSelection const& sel, QWidget* parent = NULL);
};

}

#endif
