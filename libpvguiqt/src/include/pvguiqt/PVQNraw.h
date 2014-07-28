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
	static bool show_unique_values(Picviz::PVView_sp& view, PVRush::PVNraw const& nraw, PVCol c, Picviz::PVSelection const& sel, QWidget* parent = NULL);
	static bool show_count_by(Picviz::PVView_sp& view, PVRush::PVNraw const& nraw, PVCol col1, PVCol col2, Picviz::PVSelection const& sel, QWidget* parent = NULL);
	static bool show_sum_by(Picviz::PVView_sp& view, PVRush::PVNraw const& nraw, PVCol col1, PVCol col2, Picviz::PVSelection const& sel, QWidget* parent = NULL);
	static bool show_max_by(Picviz::PVView_sp& view, PVRush::PVNraw const& nraw, PVCol col1, PVCol col2, Picviz::PVSelection const& sel, QWidget* parent = NULL);
	static bool show_min_by(Picviz::PVView_sp& view, PVRush::PVNraw const& nraw, PVCol col1, PVCol col2, Picviz::PVSelection const& sel, QWidget* parent = NULL);
	static bool show_avg_by(Picviz::PVView_sp& view, PVRush::PVNraw const& nraw, PVCol col1, PVCol col2, Picviz::PVSelection const& sel, QWidget* parent = NULL);
};

}

#endif
