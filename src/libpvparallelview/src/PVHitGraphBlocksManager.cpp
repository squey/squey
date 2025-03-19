//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/squey_intrin.h>

#include <squey/PVSelection.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVHitGraphBlocksManager.h>
#include <pvparallelview/PVHitGraphSSEHelpers.h>

#include <cassert>
#include <iostream>

inline static uint32_t y_to_block_idx(const uint32_t y, const uint32_t zoom)
{
	return y >> (32 - zoom);
}

inline static uint32_t y_to_idx_in_buffer(const uint32_t y, const uint32_t zoom)
{
	return y >> (32 - (zoom + PARALLELVIEW_ZZT_BBITS));
}

inline static uint32_t
y_to_idx_in_red_buffer(const uint32_t y, const uint32_t zoom, const double alpha)
{
	return ((double)y_to_idx_in_buffer(y, zoom)) * alpha;
}

PVParallelView::PVHitGraphBlocksManager::PVHitGraphBlocksManager(
    const uint32_t* col_scaled,
    const PVRow nrows,
    uint32_t nblocks,
    Squey::PVSelection const& layer_sel,
    Squey::PVSelection const& sel)
    : _data_z0(PARALLELVIEW_ZT_BBITS, 1)
    , _data(PARALLELVIEW_ZZT_BBITS, nblocks)
    , _layer_sel(layer_sel)
    , _sel(sel)
    , _data_params(col_scaled, nrows, 0, -1, PARALLELVIEW_ZT_BBITS, 0.5, 0, nblocks)
{
}

bool PVParallelView::PVHitGraphBlocksManager::change_and_process_view(const uint32_t y_min,
                                                                      const int zoom,
                                                                      double alpha)
{
	const uint32_t block_idx = y_to_block_idx(y_min, zoom);
	const uint32_t y_min_block = block_idx << (32 - zoom);

	const int32_t y_min_idx_in_red_buffer = y_to_idx_in_red_buffer(y_min_block, zoom, alpha);
	const int32_t last_y_min_idx_in_red_buffer = y_to_idx_in_red_buffer(last_y_min(), zoom, alpha);
	int32_t blocks_shift = (last_y_min_idx_in_red_buffer - y_min_idx_in_red_buffer) /
	                       ((int)((double)(size_block()) * alpha));

	// This is done because, at the original zoom, a reduction over 10 bits is
	// done
	if ((alpha == 0.5) && (zoom == 0)) {
		alpha = 1.0;
		blocks_shift = 0;
	}
	if (zoom == last_zoom() && alpha == last_alpha()) {
		_data_params.alpha = alpha;

		if (blocks_shift == 0) {
			return false;
		}

		// Translation
		//
		_data_params.nbits = nbits();

		if (abs(blocks_shift) >= (int)nblocks()) {
			// Reprocess all
			_data_params.y_min = y_min_block;
			_data_params.block_start = 0;
			_data_params.nblocks = full_view() ? 1 : nblocks();
			process_all_buffers();
			return true;
		}

		// Left or right shift blocks
		shift_blocks(blocks_shift, alpha);

		// Compute empty blocks
		if (blocks_shift > 0) {
			// We did a shift on the right. Let's process the first missing blocks.
			_data_params.y_min = y_min_block;
			_data_params.block_start = 0;
			_data_params.nblocks = blocks_shift;
		} else {
			// The shift was on the left, so let's process the last missing blocks.
			const int abs_blocks_shift = -blocks_shift;
			_data_params.block_start = nblocks() - abs_blocks_shift;
			_data_params.y_min = (block_idx + _data_params.block_start) << (32 - zoom);
			_data_params.nblocks = abs_blocks_shift;
		}

		_data.process_all_buffers(_data_params, _layer_sel, _sel);

		// Set last params to the full block range
		// (in case a reprocessing will be necessary)
		_data_params.y_min = y_min_block;
		_data_params.block_start = 0;
		_data_params.nblocks = nblocks();

		return true;
	}

	_data_params.alpha = alpha;
	_data_params.zoom = zoom;
	_data_params.y_min = y_min_block;
	_data_params.block_start = 0;
	_data_params.nblocks = full_view() ? 1 : nblocks();
	_data_params.nbits = nbits();

	process_all_buffers();

	return true;
}

void PVParallelView::PVHitGraphBlocksManager::process_buffer_all()
{
	if (full_view()) {
		_data_z0.buffer_all().set_zero();
		_data_z0.process_buffer_all(_data_params);
	} else {
		_data.buffer_all().set_zero();
		_data.process_buffer_all(_data_params);
	}
}

void PVParallelView::PVHitGraphBlocksManager::process_buffer_selected()
{
	if (full_view()) {
		_data_z0.buffer_selected().set_zero();
		_data_z0.process_buffer_selected(_data_params, _sel);
	} else {
		_data.buffer_selected().set_zero();
		_data.process_buffer_selected(_data_params, _sel);
	}
}

void PVParallelView::PVHitGraphBlocksManager::process_all_buffers()
{
	if (full_view()) {
		_data_z0.set_zero();
		_data_z0.process_all_buffers(_data_params, _layer_sel, _sel);
	} else {
		_data.set_zero();
		_data.process_all_buffers(_data_params, _layer_sel, _sel);
	}
}

uint32_t const* PVParallelView::PVHitGraphBlocksManager::buffer_all() const
{
	if (full_view()) {
		return _data_z0.buffer_all().buffer();
	}

	return _data.buffer_all().buffer();
}

uint32_t const* PVParallelView::PVHitGraphBlocksManager::buffer_selectable() const
{
	if (full_view()) {
		return _data_z0.buffer_selectable().buffer();
	}

	return _data.buffer_selectable().buffer();
}

uint32_t const* PVParallelView::PVHitGraphBlocksManager::buffer_selected() const
{
	if (full_view()) {
		return _data_z0.buffer_selected().buffer();
	}

	return _data.buffer_selected().buffer();
}

uint32_t PVParallelView::PVHitGraphBlocksManager::y_start() const
{
	return y_to_block_idx(_data_params.y_min, _data_params.zoom) << (32 - _data_params.zoom);
}

int PVParallelView::PVHitGraphBlocksManager::nbits() const
{
	return full_view() ? PARALLELVIEW_ZT_BBITS : PARALLELVIEW_ZZT_BBITS;
}

uint32_t PVParallelView::PVHitGraphBlocksManager::get_count_for(const uint32_t value) const
{
	const PVParallelView::PVHitGraphData& data = hgdata();

	const int zoom = last_zoom();
	const int nbits = data.nbits();
	const int idx_shift = (32 - nbits) - zoom;
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t zoom_mask = ((1ULL << zoom_shift) - 1ULL);
	const int32_t base_y = (uint64_t)(_data_params.y_min) >> zoom_shift;
	const uint32_t y_min_ref = (uint64_t)base_y << zoom_shift;

	uint64_t y = value;
	const int32_t base = (uint64_t)(y) >> zoom_shift;

	int p = base - base_y;
	if ((p < 0) || (p >= (int)data.nblocks())) {
		return 0;
	}

	y = (y - y_min_ref) * last_alpha();

	const uint32_t idx = ((uint32_t)(y & zoom_mask)) >> idx_shift;
	return data.buffer_all().buffer()[idx];
}

simde__m128i PVParallelView::PVHitGraphBlocksManager::get_count_for(simde__m128i value) const
{
	const PVParallelView::PVHitGraphData& data = hgdata();

	const int zoom = last_zoom();
	const int nbits = data.nbits();
	const int idx_shift = (32 - nbits) - zoom;
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t zoom_mask = ((1ULL << zoom_shift) - 1ULL);
	const int32_t base_y = (uint64_t)(_data_params.y_min) >> zoom_shift;
	const uint32_t y_min_ref = (uint64_t)base_y << zoom_shift;

	const simde__m128i base_y_sse = simde_mm_set1_epi32(base_y);
	const simde__m256d alpha_sse = simde_mm256_set1_pd(last_alpha());

	const simde__m128i zoom_mask_sse = simde_mm_set1_epi32(zoom_mask);
	const simde__m128i y_min_ref_sse = simde_mm_set1_epi32(y_min_ref);

	const simde__m128i base_sse = simde_mm_srli_epi32(value, zoom_shift);
	const simde__m128i p_sse = simde_mm_sub_epi32(base_sse, base_y_sse);

	const simde__m128i res_sse =
	    simde_mm_andnot_si128(
			simde_mm_cmplt_epi32(p_sse, simde_mm_setzero_si128()),
	        simde_mm_cmplt_epi32(p_sse, simde_mm_set1_epi32(data.nblocks()))
		);

	if (simde_mm_test_all_zeros(res_sse, simde_mm_set1_epi32(0xFFFFFFFFU))) {
		return simde_mm_setzero_si128();
	}

	const simde__m128i idx_sse = PVParallelView::PVHitGraphSSEHelpers::buffer_offset_from_y_sse(
	    value, p_sse, y_min_ref_sse, alpha_sse, zoom_mask_sse, idx_shift, zoom_shift, nbits);

	const uint32_t* buffer = data.buffer_all().buffer();

	// Waiting for gather support in AVX2...
	return simde_mm_set_epi32(
	    buffer[simde_mm_extract_epi32(idx_sse, 3)], buffer[simde_mm_extract_epi32(idx_sse, 2)],
	    buffer[simde_mm_extract_epi32(idx_sse, 1)], buffer[simde_mm_extract_epi32(idx_sse, 0)]);
}

void PVParallelView::PVHitGraphBlocksManager::shift_blocks(const int blocks_shift,
                                                           const double alpha)
{
	if (blocks_shift > 0) {
		hgdata().shift_right(blocks_shift, alpha);
	} else {
		hgdata().shift_left(-blocks_shift, alpha);
	}
}

PVParallelView::PVHitGraphData& PVParallelView::PVHitGraphBlocksManager::hgdata()
{
	if (full_view()) {
		return _data_z0;
	} else {
		return _data;
	}
}

PVParallelView::PVHitGraphData const& PVParallelView::PVHitGraphBlocksManager::hgdata() const
{
	if (full_view()) {
		return _data_z0;
	} else {
		return _data;
	}
}

uint32_t PVParallelView::PVHitGraphBlocksManager::get_max_count_all() const
{
	if (full_view()) {
		return _data_z0.buffer_all().get_max_count();
	}
	return _data.buffer_all().get_zoomed_max_count(last_alpha());
}

uint32_t PVParallelView::PVHitGraphBlocksManager::get_max_count_selected() const
{
	if (full_view()) {
		return _data_z0.buffer_selected().get_max_count();
	}
	return _data.buffer_selected().get_zoomed_max_count(last_alpha());
}
