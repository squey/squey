/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2018
 */

#include "inendi/PVRangeSubSampler.h"

#include <pvcop/db/array.h>
#include <pvcop/db/algo.h>
#include <pvcop/types/datetime_us.h>
#include <pvkernel/core/inendi_bench.h> // for BENCH_END, BENCH_START
#include <inendi/PVPlotted.h>

#include <boost/date_time/posix_time/posix_time.hpp>

#include <numeric>
#include <math.h>

Inendi::PVRangeSubSampler::PVRangeSubSampler(
    const pvcop::db::array& time,
    const std::vector<pvcop::core::array<value_type>>& timeseries,
    const PVRush::PVNraw& nraw,
    const pvcop::db::selection& sel,
    size_t sampling_count /*= 2048*/)
    : _time(time), _timeseries(timeseries), _nraw(nraw), _sel(sel), _minmax()
{
	set_sampling_count(
	    sampling_count); // should be the number of horizontal visible pixels in the plot

	BENCH_START(sort);

	if (not _time.is_sorted()) { // FIXME
		_sorted_indexes = time.parallel_sort();
		_sort = _sorted_indexes.to_core_array();
	}

	BENCH_END(sort, "sort", _time.size(), sizeof(uint64_t), _time.size(), sizeof(uint64_t));

	_minmax = pvcop::db::algo::minmax(_time);
	_last_params = SamplingParams(0, 0, _minmax, 0, 0);
}

void Inendi::PVRangeSubSampler::set_sampling_count(size_t sampling_count)
{
	_sampling_count = sampling_count;
	_reset = true;
}

pvcop::db::array Inendi::PVRangeSubSampler::ratio_to_minmax(zoom_f ratio1, zoom_f ratio2) const
{

	return _time.ratio_to_minmax(ratio1, ratio2, _minmax);
}

std::pair<Inendi::PVRangeSubSampler::zoom_f, Inendi::PVRangeSubSampler::zoom_f>
Inendi::PVRangeSubSampler::minmax_to_ratio(const pvcop::db::array& minmax) const
{
	return _time.minmax_to_ratio(minmax, _minmax);
}

void Inendi::PVRangeSubSampler::subsample(zoom_f first_ratio,
                                          zoom_f last_ratio,
                                          zoom_f min_ratio /*= 0*/,
                                          zoom_f max_ratio /*= 0*/)
{
	const value_type min = min_ratio * std::numeric_limits<value_type>::max();
	const value_type max = max_ratio * std::numeric_limits<value_type>::max();

	subsample(ratio_to_minmax(first_ratio, last_ratio), min, max);
}

void Inendi::PVRangeSubSampler::subsample(const pvcop::db::array& minmax,
                                          uint32_t min /*= 0*/,
                                          uint32_t max /*= 0*/)
{
	auto[first, past_end] = _time.equal_range(minmax, _sorted_indexes);
	subsample(first, past_end - 1, minmax, min, max);
}

void Inendi::PVRangeSubSampler::subsample(size_t first,
                                          size_t last,
                                          const pvcop::db::array& minmax,
                                          uint32_t min /*= 0*/,
                                          uint32_t max /*= 0*/)
{
	if (last == 0) {
		last = _time.size() - 1;
	}
	if (max == 0) {
		max = std::numeric_limits<uint32_t>::max();
	}

	// Resample every selected timeseries if params have changed
	if (SamplingParams(first, last, minmax.copy(), min, max) != _last_params) {
		_timeseries_to_subsample =
		    std::vector<size_t>(_selected_timeseries.begin(), _selected_timeseries.end());
	}
	_last_params = SamplingParams(first, last, minmax.copy(), min, max);

	pvlogger::info() << "PVRangeSubSampler::subsample(first:" << first << ", last:" << last
	                 << ",minmax: " << minmax.at(0) << " .. " << minmax.at(1) << ", min:" << min
	                 << ", max:" << max << ", _sampling_count:" << _sampling_count
	                 << ", _reset:" << _reset
	                 << ", _timeseries_to_subsample.size():" << _timeseries_to_subsample.size()
	                 << ")\n";

	if (_reset) {
		allocate_internal_structures(); // start and end are not included
		_reset = false;
	}

	_time.histogram(first, last, minmax, _sorted_indexes, _histogram);
	compute_ranges_average(first, last, min, max);

	_subsampled.emit();
}

void Inendi::PVRangeSubSampler::set_selected_timeseries(
    const std::unordered_set<size_t>& selected_timeseries)
{
	_timeseries_to_subsample.clear();
	std::copy_if(selected_timeseries.begin(), selected_timeseries.end(),
	             std::back_inserter(_timeseries_to_subsample), [this](size_t index) {
		             return _selected_timeseries.find(index) == _selected_timeseries.end();
		         });
	_selected_timeseries = selected_timeseries;
}

void Inendi::PVRangeSubSampler::resubsample()
{
	_timeseries_to_subsample =
	    std::vector<size_t>(_selected_timeseries.begin(), _selected_timeseries.end());

	subsample(_last_params.first, _last_params.last, _last_params.minmax, _last_params.min,
	          _last_params.max);
}

void Inendi::PVRangeSubSampler::resubsample(const std::unordered_set<size_t>& timeseries)
{
	_timeseries_to_subsample.clear();
	std::copy(timeseries.begin(), timeseries.end(), std::back_inserter(_timeseries_to_subsample));

	subsample(_last_params.first, _last_params.last, _last_params.minmax, _last_params.min,
	          _last_params.max);
}

void Inendi::PVRangeSubSampler::allocate_internal_structures()
{
	assert(_sampling_count >= 2);

	// range values count
	_histogram = std::vector<size_t>(_sampling_count);

	assert(_timeseries.size() > 0);
	assert(_histogram.size() > 0);

	// matrix of average values
	_avg_matrix.resize(_timeseries.size());
	for (auto& vec : _avg_matrix) {
		vec.resize(_histogram.size());
	}

	_timeseries_to_subsample.clear();
	std::copy(_selected_timeseries.begin(), _selected_timeseries.end(),
	          std::back_inserter(_timeseries_to_subsample));
}

bool Inendi::PVRangeSubSampler::valid() const
{
	return _avg_matrix.size() > 0;
}

void Inendi::PVRangeSubSampler::compute_ranges_average(size_t first,
                                                       size_t /*last*/,
                                                       size_t min,
                                                       size_t max)
{
	BENCH_START(computing_average);

	// Remove invalid values from selection
	const pvcop::db::selection& valid_sel = _time.valid_selection(_sel);

#pragma omp parallel for
	for (size_t k = 0; k < _timeseries_to_subsample.size(); k++) {

		const size_t i = _timeseries_to_subsample[k];

		size_t start = first;
		size_t end = first;
		const pvcop::core::array<value_type>& timeserie = _timeseries[i];
		const pvcop::db::selection& ts_valid_sel =
		    _nraw.column(PVCol(i)).valid_selection(valid_sel);
		for (size_t j = 0; j < _histogram.size(); j++) {
			const size_t values_count = _histogram[j];
			end += values_count;
			size_t selected_values_count = 0;
			uint64_t sum = 0;
			for (size_t k = start; k < end; k++) {
				auto v = not _sort ? k : _sort[k];
				if (ts_valid_sel[v]) {
					selected_values_count++;
					sum += (std::numeric_limits<value_type>::max() - timeserie[v]);
				}
			}
			start = end;
			if (selected_values_count == 0) {
				_avg_matrix[i][j] = no_value; // no value in range
			} else {
				const uint64_t raw_value = sum / selected_values_count;
				if (min != 0 and raw_value < min) { // underflow
					_avg_matrix[i][j] = underflow_value;
				} else if (raw_value > max) { // overflow
					_avg_matrix[i][j] = overflow_value;
				} else {
					_avg_matrix[i][j] = (display_type)((zoom_f(raw_value - min) / (max - min)) *
					                                   display_type_max_val); // nominal value
				}
			}
		}
	}

	BENCH_END(computing_average, "computing_average", _time.size(), sizeof(uint64_t),
	          _sampling_count, sizeof(uint64_t));
}
