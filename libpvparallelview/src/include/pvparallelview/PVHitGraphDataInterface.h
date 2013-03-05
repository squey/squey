#ifndef PVPARALLELVIEW_PVHITGRAPHDATAINTERFACE_H
#define PVPARALLELVIEW_PVHITGRAPHDATAINTERFACE_H

#include <pvbase/types.h>

#include <pvparallelview/PVHitGraphCommon.h>
#include <pvparallelview/PVHitGraphBuffer.h>

#include <boost/noncopyable.hpp>

#include <cstddef>
#include <cstdint>

namespace Picviz {
class PVSelection;
}

namespace PVParallelView {

// Forward declarations
class PVZoneTree;

class PVHitGraphDataInterface: boost::noncopyable
{
public:
	PVHitGraphDataInterface();
	virtual ~PVHitGraphDataInterface();

public:
	struct ProcessParams
	{
		ProcessParams(PVZoneTree const& zt_, uint32_t const* col_plotted_, PVRow const nrows_, uint32_t const y_min_, int const zoom_, int const block_start_, int const nblocks_):
			zt(zt_),
			col_plotted(col_plotted_),
			nrows(nrows_),
			y_min(y_min_),
			zoom(zoom_),
			block_start(block_start_),
			nblocks(nblocks_)
		{ }

		PVZoneTree const& zt;
		uint32_t const* col_plotted;
		PVRow const nrows;
		uint32_t const y_min;
		int const zoom;
		int const block_start;
		int const nblocks;
	};

public:
	virtual void process_all(ProcessParams const& params) = 0;
	virtual void process_sel(ProcessParams const& params, Picviz::PVSelection const& sel) = 0;
	virtual void process_allandsel(ProcessParams const& params, Picviz::PVSelection const& sel);

public:
	void shift_left(int n);
	void shift_right(int n);

public:
	PVHitGraphBuffer const& buffer_all() const { return _buf_all; }
	PVHitGraphBuffer const& buffer_sel() const { return _buf_sel; }

	PVHitGraphBuffer& buffer_all() { return _buf_all; }
	PVHitGraphBuffer& buffer_sel() { return _buf_sel; }

private:
	PVHitGraphBuffer _buf_all;
	PVHitGraphBuffer _buf_sel;
};

}

#endif
