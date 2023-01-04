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

#ifndef PVPARALLELVIEW_PVHITGRAPHBLOCKSMANAGER_H
#define PVPARALLELVIEW_PVHITGRAPHBLOCKSMANAGER_H

#include <pvkernel/core/inendi_intrin.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVHitGraphData.h>

namespace Inendi
{
class PVSelection;
} // namespace Inendi

namespace PVParallelView
{

class PVHitGraphBlocksManager : boost::noncopyable
{
  protected:
	typedef PVHitGraphData::ProcessParams DataProcessParams;

  public:
	PVHitGraphBlocksManager(const uint32_t* col_plotted,
	                        const PVRow nrows,
	                        uint32_t nblocks,
	                        Inendi::PVSelection const& layer_sel,
	                        Inendi::PVSelection const& sel);

  public:
	bool change_and_process_view(const uint32_t y_min, const int zoom, double alpha);
	void process_buffer_all();
	void process_buffer_selected();
	void process_all_buffers();

  public:
	uint32_t const* buffer_all() const;
	uint32_t const* buffer_selectable() const;
	uint32_t const* buffer_selected() const;

	uint32_t y_start() const;
	int nbits() const;
	inline uint32_t nblocks() const { return _data.nblocks(); }

	inline uint32_t size_int() const { return hgdata().size_int(); }

	inline const uint32_t* get_plotted() const { return _data_params.col_plotted; }
	inline PVRow get_nrows() const { return _data_params.nrows; }

	uint32_t get_count_for(const uint32_t value) const;
	__m128i get_count_for(__m128i value) const;

	uint32_t get_max_count_all() const;
	uint32_t get_max_count_selected() const;

  public:
	inline int last_zoom() const { return _data_params.zoom; }
	inline int last_nbits() const { return _data_params.nbits; }
	inline double last_alpha() const { return _data_params.alpha; }
	inline uint32_t last_y_min() const { return _data_params.y_min; }
	inline uint32_t size_block() const { return _data.size_block(); }

	PVHitGraphData const& hgdata() const;

  protected:
	inline bool full_view() const
	{
		return (_data_params.zoom == 0) && (_data_params.alpha == 1.0);
	}
	PVHitGraphData& hgdata();

	void shift_blocks(int blocks_shift, const double alpha);

  protected:
	PVHitGraphData _data_z0; // Data for initial zoom (with 10-bit precision)
	PVHitGraphData _data;

	Inendi::PVSelection const& _layer_sel;
	Inendi::PVSelection const& _sel;

	DataProcessParams _data_params;
};
} // namespace PVParallelView

#endif
