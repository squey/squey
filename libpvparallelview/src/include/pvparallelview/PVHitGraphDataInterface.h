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
	PVHitGraphDataInterface(uint32_t nbits, uint32_t nblocks);
	virtual ~PVHitGraphDataInterface();

public:
	struct ProcessParams
	{
		ProcessParams(uint32_t const* col_plotted_, PVRow const nrows_, uint32_t const y_min_, int const zoom_, const double &alpha_, int const block_start_, int const nblocks_):
			col_plotted(col_plotted_),
			nrows(nrows_),
			y_min(y_min_),
			zoom(zoom_),
			alpha(alpha_),
			block_start(block_start_),
			nblocks(nblocks_)
		{ }

		uint32_t const* col_plotted;
		PVRow nrows;
		uint32_t y_min;
		int zoom;
		double alpha;
		int block_start;
		int nblocks;
	};

public:
	virtual void process_bg(ProcessParams const& params) = 0;
	virtual void process_sel(ProcessParams const& params, Picviz::PVSelection const& sel) = 0;
	virtual void process_all(ProcessParams const& params, Picviz::PVSelection const& sel);

public:
	void shift_left(const uint32_t nblocks, const double alpha);
	void shift_right(const uint32_t nblocks, const double alpha);

public:
	PVHitGraphBuffer const& buffer_all() const { return _buf_all; }
	PVHitGraphBuffer const& buffer_sel() const { return _buf_sel; }

	PVHitGraphBuffer& buffer_all() { return _buf_all; }
	PVHitGraphBuffer& buffer_sel() { return _buf_sel; }

	void set_zero()
	{
		buffer_all().set_zero();
		buffer_sel().set_zero();
	}

public:
	inline uint32_t nbits() const { return buffer_all().nbits(); }
	inline uint32_t size_block() const { return buffer_all().size_block(); }
	inline uint32_t size_int() const { return buffer_all().size_int(); }
	inline uint32_t nblocks() const { return buffer_all().nblocks(); }

private:
	PVHitGraphBuffer _buf_all;
	PVHitGraphBuffer _buf_sel;
};

}

#endif
