//
// MIT License
//
// © Squey, 2026
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

#include <pvparallelview/PVSeriesSampleRows.h>

#include <squey/PVRangeSubSampler.h>

#include <algorithm>

namespace PVParallelView
{

void PVSeriesSampleRows::project(Squey::PVRangeSubSampler const& rss,
                                 std::vector<PVSeriesView::SerieDrawInfo> const& draw_order,
                                 PVSeriesView::DrawMode draw_mode,
                                 int samples_count,
                                 int height,
                                 std::vector<int16_t>& rows)
{
	using PVRSS = Squey::PVRangeSubSampler;

	const int series_count = int(draw_order.size());
	rows.resize(size_t(series_count) * samples_count);

#pragma omp parallel for schedule(static) if (series_count > 1)
	for (int s = 0; s < series_count; ++s) {
		auto const& serie_data = rss.sampled_timeserie(draw_order[s].dataIndex);
		int16_t* serie_rows = rows.data() + size_t(s) * samples_count;

		for (int j = 0; j < samples_count; ++j) {
			const PVRSS::display_type vertex = serie_data[j];
			if (PVRSS::display_match(vertex, PVRSS::overflow_value)) {
				// Just above the top edge, where QPainter used to put it so that a
				// segment reaching it still crosses the visible rows.
				serie_rows[j] = -1;
			} else if (PVRSS::display_match(vertex, PVRSS::underflow_value)) {
				serie_rows[j] = int16_t(height);
			} else if (PVRSS::display_match(vertex, PVRSS::no_value)) {
				serie_rows[j] = no_row;
			} else {
				serie_rows[j] =
				    int16_t((height - 1) - int(vertex) * (height - 1) / PVRSS::display_type_max_val);
			}
		}

		if (draw_mode != PVSeriesView::DrawMode::LinesAlways) {
			continue;
		}

		// LinesAlways joins across the gaps and reaches both edges, so fill the holes in
		// with the straight line the reference renderer would have drawn over them.
		// Everything downstream then sees an ordinary uninterrupted serie.
		int previous = -1;
		for (int j = 0; j < samples_count; ++j) {
			if (serie_rows[j] == no_row) {
				continue;
			}
			if (previous < 0) {
				std::fill(serie_rows, serie_rows + j, serie_rows[j]);
			} else if (previous != j - 1) {
				const int span = j - previous;
				const int from = serie_rows[previous];
				const int to = serie_rows[j];
				for (int k = previous + 1; k < j; ++k) {
					serie_rows[k] = int16_t(from + (to - from) * (k - previous) / span);
				}
			}
			previous = j;
		}
		if (previous >= 0) {
			std::fill(serie_rows + previous + 1, serie_rows + samples_count, serie_rows[previous]);
		}
	}
}

} // namespace PVParallelView
