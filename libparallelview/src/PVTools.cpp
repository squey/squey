#include <pvkernel/core/general.h>
#include <pvparallelview/PVTools.h>

#include <limits.h>

void PVParallelView::PVTools::norm_int_plotted(Picviz::PVPlotted::plotted_table_t const& plotted, plotted_int_t& res, PVCol ncols)
{
	// Here, we make every row starting on a 16-byte boundary
	PVRow nrows = plotted.size()/ncols;
	PVRow nrows_aligned = ((nrows+3)/4)*4;
	size_t dest_size = nrows_aligned*ncols;
	res.reserve(dest_size);
	for (PVCol c = 0; c < ncols; c++) {
		for (PVRow r = 0; r < nrows; r++) {
			res.push_back((uint32_t) ((double)plotted[c*nrows+r] * (double)UINT_MAX));
		}
		for (PVRow r = nrows; r < nrows_aligned; r++) {
			res.push_back(0);
		}
	}
}
