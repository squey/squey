#ifndef PVPARALLELVIEW_PVHITGRAPHBLOCKSMANAGER_H
#define PVPARALLELVIEW_PVHITGRAPHBLOCKSMANAGER_H

#include <pvparallelview/PVHitGraphData.h>

namespace Picviz {
class PVSelection;
}

namespace PVParallelView {

namespace __impl {
class PVHitGraphBlocksManager;
}

class PVHitGraphBlocksManager: boost::noncopyable
{
	friend class __impl::PVHitGraphBlocksManager;

protected:
	typedef PVHitGraphData::ProcessParams DataProcessParams ;

public:
	PVHitGraphBlocksManager(PVZoneTree const& zt, const uint32_t* col_plotted, const PVRow nrows, uint32_t nblocks, Picviz::PVSelection const& sel);

public:
	bool change_and_process_view(const uint32_t y_min, const int zoom, const float alpha);
	void process_bg();
	void process_sel();
	void process_all();

public:
	uint32_t const* buffer_bg() const;
	uint32_t const* buffer_sel() const;

	uint32_t const y_start() const;
	inline uint32_t nblocks() const { return _data.nblocks(); }

protected:
	inline int last_zoom() const { return _data_params.zoom; }
	inline float last_alpha() const { return _data_params.alpha; }
	inline float last_y_min() const { return _data_params.y_min; }
	inline uint32_t size_block() const { return _data.size_block(); }
	inline bool full_view() const { return (_data_params.zoom == 0) && (_data_params.alpha == 1.0f); }
	PVHitGraphData& hgdata();
	PVHitGraphData const& hgdata() const;

	void shift_blocks(int blocks_shift, const float alpha);

protected:
	PVHitGraphData _data_z0; // Data for initial zoom (with 10-bit precision)
	PVHitGraphData _data;

	Picviz::PVSelection const& _sel;

	DataProcessParams _data_params;
};

}

#endif
