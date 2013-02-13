#ifndef PVPARALLELVIEW_PVHITGRAPHDATAOMP_H
#define PVPARALLELVIEW_PVHITGRAPHDATAOMP_H

#include <pvparallelview/PVHitGraphBuffer.h>
#include <pvparallelview/PVHitGraphDataInterface.h>

namespace PVParallelView {

class PVHitGraphDataZT: public PVHitGraphDataInterface
{
public:
	void process_all(PVZoneTree const& zt, uint32_t const* col_plotted, PVRow const nrows, uint32_t const y_min, int const zoom, int const block_start, int const nblocks) override;
	void process_sel(PVZoneTree const& zt, uint32_t const* col_plotted, PVRow const nrows, uint32_t const y_min, int const zoom, int const block_start, int const nblocks, Picviz::PVSelection const& sel) override;
};

}

#endif
