#ifndef PVPARALLELVIEW_PVHITGRAPHDATAOMP_H
#define PVPARALLELVIEW_PVHITGRAPHDATAOMP_H

#include <pvparallelview/PVHitGraphBuffer.h>
#include <pvparallelview/PVHitGraphDataInterface.h>

namespace PVParallelView {

class PVHitGraphDataOMP: public PVHitGraphDataInterface
{
public:
	PVHitGraphDataOMP(uint32_t nbits, uint32_t nblocks);

protected:
	void process_all(ProcessParams const& params, PVHitGraphBuffer& buf) const override;
	void process_sel(ProcessParams const& params, PVHitGraphBuffer& buf, Picviz::PVSelection const& sel) const override;

public:
	struct omp_ctx_t
	{
		omp_ctx_t(uint32_t size); // size is number of integers
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
	mutable omp_ctx_t _omp_ctx;
};

}

#endif
