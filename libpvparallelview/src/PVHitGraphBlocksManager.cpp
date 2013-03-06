#include <pvparallelview/common.h>
#include <pvparallelview/PVHitGraphBlocksManager.h>

#include <cassert>

inline static uint32_t masked_y_min(const uint32_t ymin, const uint32_t zoom)
{
	return ymin >> (32-zoom);
}

PVParallelView::PVHitGraphBlocksManager::PVHitGraphBlocksManager(PVZoneTree const& zt, const uint32_t* col_plotted, const PVRow nrows, uint32_t nblocks, Picviz::PVSelection const& sel):
	_data_z0(PARALLELVIEW_ZT_BBITS, 1),
	_data(PARALLELVIEW_ZZT_BBITS, nblocks),
	_sel(sel),
	_data_params(zt, col_plotted, nrows, 0, -1, 0, nblocks),
	_last_alpha(0.5f)
{
}

bool PVParallelView::PVHitGraphBlocksManager::change_and_process_view(const uint32_t y_min, const int zoom, const float alpha)
{
	if (last_zoom() != zoom) {
		// Reprocess everything
		_data_params.zoom = zoom;
		_data_params.y_min = y_min;
		_data_params.block_start = 0;
		_data_params.nblocks = full_view() ? 1 : nblocks();

		process_allandsel();
		return true;
	}

	const uint32_t y_min_masked = masked_y_min(y_min, zoom);
	const uint32_t last_y_min_masked = masked_y_min(last_y_min(), zoom);
	if (y_min_masked != last_y_min_masked) {
		// Translation
		
		_data_params.y_min = y_min;
		_last_alpha = alpha;

		int32_t blocks_shift = (int32_t)y_min_masked - (int32_t)last_y_min_masked;

		if (abs(blocks_shift) >= (int) nblocks()) {
			// Reprocess all
			_data_params.block_start = 0;
			_data_params.nblocks = full_view() ? 1 : nblocks();
			process_allandsel();
			return true;
		}

		// Left or right shift blocks 
		shift_blocks(blocks_shift);

		// Compute empty blocks
		if (blocks_shift > 0) {
			// We did a shift on the right. Let's process the first missing blocks.
			_data_params.block_start = 0;
			_data_params.nblocks = blocks_shift;
		}
		else {
			// The shift was on the left, so let's process the last missing blocks.
			blocks_shift = -blocks_shift;
			_data_params.block_start = nblocks() - blocks_shift;
			_data_params.nblocks = blocks_shift;
		}

		process_allandsel();
		
		// Set last params to the full block range
		// (in case a reprocessing will be necessary)
		_data_params.block_start = 0;
		_data_params.nblocks = nblocks();

		return true;
	}

	if (alpha != last_alpha()) {
		if (full_view()) {
			return false;
		}

		_last_alpha = alpha;
		_data.process_zoom_reduction(alpha);
		return true;
	}

	// Returning false means that no computation has occured.
	return false;
}

void PVParallelView::PVHitGraphBlocksManager::process_all()
{
	if (full_view()) {
		_data_z0.buffer_all().set_zero();
		_data_z0.process_all(_data_params);
	}
	else {
		assert(_last_alpha != 0.0f);
		_data.buffer_all().set_zero();
		_data.process_all(_data_params);
		_data.buffer_all().process_zoom_reduction(_last_alpha);
	}
}

void PVParallelView::PVHitGraphBlocksManager::process_sel()
{
	if (full_view()) {
		_data_z0.buffer_sel().set_zero();
		_data_z0.process_sel(_data_params, _sel);
	}
	else {
		assert(_last_alpha != 0.0f);
		_data.buffer_sel().set_zero();
		_data.process_sel(_data_params, _sel);
		_data.buffer_sel().process_zoom_reduction(_last_alpha);
	}
}

void PVParallelView::PVHitGraphBlocksManager::process_allandsel()
{
	if (full_view()) {
		_data_z0.set_zero();
		_data_z0.process_allandsel(_data_params, _sel);
	}
	else {
		assert(_last_alpha != 0.0f);
		_data.set_zero();
		_data.process_allandsel(_data_params, _sel);
		_data.process_zoom_reduction(_last_alpha);
	}
}

uint32_t const* PVParallelView::PVHitGraphBlocksManager::buffer_all() const
{
	if (full_view()) {
		return _data_z0.buffer_all().buffer();
	}

	return _data.buffer_all().zoomed_buffer();
}

uint32_t const* PVParallelView::PVHitGraphBlocksManager::buffer_sel() const
{
	if (full_view()) {
		return _data_z0.buffer_sel().buffer();
	}

	return _data.buffer_sel().zoomed_buffer();
}

uint32_t const PVParallelView::PVHitGraphBlocksManager::y_start() const
{
	return masked_y_min(_data_params.y_min, _data_params.zoom) << (32-_data_params.zoom);
}

void PVParallelView::PVHitGraphBlocksManager::shift_blocks(int blocks_shift)
{
	if (blocks_shift > 0) {
		hgdata().shift_right(blocks_shift);
	}
	else {
		hgdata().shift_left(-blocks_shift);
	}
}

PVParallelView::PVHitGraphData& PVParallelView::PVHitGraphBlocksManager::hgdata()
{
	if (full_view()) {
		return _data_z0;
	}
	else {
		return _data;
	}
}

PVParallelView::PVHitGraphData const& PVParallelView::PVHitGraphBlocksManager::hgdata() const
{
	if (full_view()) {
		return _data_z0;
	}
	else {
		return _data;
	}
}
