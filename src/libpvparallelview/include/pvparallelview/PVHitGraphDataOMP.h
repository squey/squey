/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVPARALLELVIEW_PVHITGRAPHDATAOMP_H
#define PVPARALLELVIEW_PVHITGRAPHDATAOMP_H

#include <pvparallelview/PVHitGraphBuffer.h>
#include <pvparallelview/PVHitGraphDataInterface.h>

namespace PVParallelView
{

class PVHitGraphDataOMP : public PVHitGraphDataInterface
{
  public:
	PVHitGraphDataOMP(uint32_t nbits, uint32_t nblocks);

  protected:
	void process_all(ProcessParams const& params, PVHitGraphBuffer& buf) const override;
	void process_sel(ProcessParams const& params,
	                 PVHitGraphBuffer& buf,
	                 Squey::PVSelection const& sel) const override;

  public:
	struct omp_ctx_t {
		explicit omp_ctx_t(uint32_t size); // size is number of integers
		~omp_ctx_t();

		void clear();

		int get_core_num() const { return _core_num; }

		uint32_t* get_core_buffer(int i) { return _buffers[i]; }

		uint32_t _buffer_size;
		uint32_t _core_num;
		int _block_count;
		uint32_t** _buffers;
	};

  private:
	mutable omp_ctx_t _omp_ctx;
};
} // namespace PVParallelView

#endif
