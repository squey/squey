#ifndef PVPARALLELVIW_PVZONEPROCESSING_H
#define PVPARALLELVIW_PVZONEPROCESSING_H

#include <pvkernel/core/general.h>
#include <picviz/PVPlotted.h>

namespace PVParallelView {

class PVZoneProcessing
{
public:
	PVZoneProcessing(
		Picviz::PVPlotted::uint_plotted_table_t const& plotted_,
		PVRow nrows_,
		PVCol col_a_,
		PVCol col_b_
	):
		plotted(plotted_),
		nrows(nrows_),
		col_a(col_a_),
		col_b(col_b_),
		nrows_aligned(ALIGN_SIZE(nrows, PVROW_VECTOR_ALIGNMENT))
	{ }

public:
	inline PVRow nrows() const { return _nrows; }
	inline PVCol col_a() const { return _col_a; }
	inline PVCol col_b() const { return _col_b; }
	inline PVRow nrows_aligned() const { return _nrows_aligned; }

private:
	Picviz::PVPlotted::uint_plotted_table_t const& _plotted;
	PVRow _nrows;
	PVCol _col_a;
	PVCol _col_b;
	PVRow _nrows_aligned;
};

}

#endif
