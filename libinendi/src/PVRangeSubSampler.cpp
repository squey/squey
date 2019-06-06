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
    const pvcop::db::array* split /* =  nullptr */,
    size_t sampling_count /*= 2048*/)
    : _time(time), _timeseries(timeseries), _nraw(nraw), _sel(sel), _split(split), _minmax()
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

	set_sampling_mode<SAMPLING_MODE::MEAN>();

	allocate_internal_structures();
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

#ifdef INENDI_DEVELOPER_MODE
	pvlogger::info() << "PVRangeSubSampler::subsample(first:" << first << ", last:" << last
	                 << ",minmax: " << minmax.at(0) << " .. " << minmax.at(1) << ", min:" << min
	                 << ", max:" << max << ", _sampling_count:" << _sampling_count
	                 << ", _reset:" << _reset
	                 << ", _timeseries_to_subsample.size():" << _timeseries_to_subsample.size()
	                 << ")\n";
#endif

	if (_reset) {
		allocate_internal_structures();
		_reset = false;
	}

	_time.histogram(first, last, minmax, _sorted_indexes, _histogram);
	_compute_ranges_reduction_f(first, last, min, max);

	_subsampled.emit();
	_valid = true;
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
	subsample(_last_params.first, _last_params.last, _last_params.minmax, _last_params.min,
	          _last_params.max);

	_timeseries_to_subsample =
	    std::vector<size_t>(_selected_timeseries.begin(), _selected_timeseries.end());
}

void Inendi::PVRangeSubSampler::resubsample(const std::unordered_set<size_t>& timeseries)
{
	_timeseries_to_subsample.clear();
	std::copy(timeseries.begin(), timeseries.end(), std::back_inserter(_timeseries_to_subsample));

	subsample(_last_params.first, _last_params.last, _last_params.minmax, _last_params.min,
	          _last_params.max);
}

void Inendi::PVRangeSubSampler::set_split_column(const pvcop::db::array* split)
{
	_split_count = 1;
	_split = split;
	if (_split) {
		_split_groups.~groups();
		new (&_split_groups) pvcop::db::groups();
		_split_extents.~extents();
		new (&_split_extents) pvcop::db::extents();
		_split->group(_split_groups, _split_extents);
		_split_count = _split_extents.size();
	}
	_ts_matrix.resize(_timeseries.size() * _split_count);
	for (auto& vec : _ts_matrix) {
		vec.resize(_histogram.size());
	}
}

void Inendi::PVRangeSubSampler::allocate_internal_structures()
{
	assert(_sampling_count >= 2);

	// range values count
	_histogram = std::vector<size_t>(_sampling_count);

	assert(_timeseries.size() > 0);
	assert(_histogram.size() > 0);

	// matrix of average values
	set_split_column(_split);

	_timeseries_to_subsample.clear();
	std::copy(_selected_timeseries.begin(), _selected_timeseries.end(),
	          std::back_inserter(_timeseries_to_subsample));
}

bool Inendi::PVRangeSubSampler::valid() const
{
	return _valid;
}
