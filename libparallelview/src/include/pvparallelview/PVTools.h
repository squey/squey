#ifndef PVPARALLELVIEW_TOOLS_H
#define PVPARALLELVIEW_TOOLS_H

#include <pvparallelview/common.h>
#include <picviz/PVPlotted.h>

namespace PVParallelView {

namespace PVTools {

void norm_int_plotted(Picviz::PVPlotted::plotted_table_t const& plotted, plotted_int_t& res, PVCol ncols);

}

}

#endif
