#ifndef PVPARALLELVIEW_PVHITGRAPHBLOCKSMANAGER_H
#define PVPARALLELVIEW_PVHITGRAPHBLOCKSMANAGER_H

#include <pvkernel/core/picviz_intrin.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVHitGraphData.h>

namespace Picviz {
class PVSelection;
}

namespace PVParallelView {

class PVHitGraphBlocksManager: boost::noncopyable
{
protected:
	typedef PVHitGraphData::ProcessParams DataProcessParams ;

public:
	PVHitGraphBlocksManager(const uint32_t* col_plotted, const PVRow nrows, uint32_t nblocks, Picviz::PVSelection& layer_sel, Picviz::PVSelection const& sel);

public:
	bool change_and_process_view(const uint32_t y_min, const int zoom, double alpha);
	void process_bg();
	void process_sel();
	void process_all();

public:
	void set_layer_sel(const Picviz::PVSelection &sel);

public:
	uint32_t const* buffer_bg() const;
	uint32_t const* buffer_sel() const;

	uint32_t y_start() const;
	int nbits() const;
	inline uint32_t nblocks() const { return _data.nblocks(); }

	inline uint32_t size_int() const { return hgdata().size_int(); }

	inline const uint32_t* get_plotted() const { return _data_params.col_plotted; }
	inline PVRow get_nrows() const { return _data_params.nrows; }

	uint32_t get_count_for(const uint32_t value) const;
	__m128i  get_count_for(__m128i value) const;

	uint32_t get_max_count_all() const;
	uint32_t get_max_count_sel() const;

public:
	inline int last_zoom() const { return _data_params.zoom; }
	inline int last_nbits() const { return _data_params.nbits; }
	inline double last_alpha() const { return _data_params.alpha; }
	inline uint32_t last_y_min() const { return _data_params.y_min; }
	inline uint32_t size_block() const { return _data.size_block(); }

	PVHitGraphData const& hgdata() const;

protected:
	inline bool full_view() const { return (_data_params.zoom == 0) && (_data_params.alpha == 1.0); }
	PVHitGraphData& hgdata();

	void shift_blocks(int blocks_shift, const double alpha);

protected:
	PVHitGraphData _data_z0; // Data for initial zoom (with 10-bit precision)
	PVHitGraphData _data;

	Picviz::PVSelection& _layer_sel;
	Picviz::PVSelection const& _sel;

	DataProcessParams _data_params;
};

}

#endif
