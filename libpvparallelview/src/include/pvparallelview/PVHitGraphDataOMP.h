#ifndef PVPARALLELVIEW_PVHITGRAPHDATAOMP_H
#define PVPARALLELVIEW_PVHITGRAPHDATAOMP_H

#include <pvparallelview/PVHitGraphBuffer.h>
#include <pvparallelview/PVHitGraphDataInterface.h>

namespace PVParallelView {

class PVHitGraphDataOMP: public PVHitGraphDataInterface
{
public:
	void process_all(PVZoneTree const& zt, uint32_t const* col_plotted, PVRow const nrows, uint32_t const y_min, int const zoom, int const block_start, int const nblocks) override;
	void process_sel(PVZoneTree const& zt, uint32_t const* col_plotted, PVRow const nrows, uint32_t const y_min, int const zoom, int const block_start, int const nblocks, Picviz::PVSelection const& sel) override;

public:
	struct omp_ctx_t
	{
		omp_ctx_t(uint32_t size = PVHitGraphBuffer::SIZE_BLOCK*PVHitGraphBuffer::NBLOCKS);
		~omp_ctx_t();

		void clear();

		int get_core_num() const
		{
			return _core_num;
		}

		uint32_t *get_core_buffer(int i)
		{
			return _buffers[i];
		}

		uint32_t   _buffer_size;
		uint32_t   _core_num;
		int        _block_count;
		uint32_t **_buffers;
	};

private:
	omp_ctx_t _omp_ctx;
};

}

#endif
