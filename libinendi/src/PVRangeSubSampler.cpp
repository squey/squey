/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2018
 */

#include "inendi/PVRangeSubSampler.h"

#include <pvcop/db/array.h>
#include <pvcop/types/datetime_us.h>
#include <pvkernel/core/inendi_bench.h> // for BENCH_END, BENCH_START
#include <inendi/PVPlotted.h>

#include <boost/date_time/posix_time/posix_time.hpp>

#include <numeric>
#include <math.h>

Inendi::PVRangeSubSampler::PVRangeSubSampler(
    const pvcop::db::array& time,
    const std::vector<pvcop::core::array<value_type>>& timeseries,
    const pvcop::db::selection& sel,
    size_t sampling_count /*= 2048*/)
    : _time(time), _timeseries(timeseries), _sel(sel), _minmax()
{
	set_sampling_count(
	    sampling_count); // should be the number of horizontal visible pixels in the plot

	BENCH_START(sort);

	if (not _time.is_sorted()) {
		_sorted_indexes = time.parallel_sort();
		_sort = _sorted_indexes.to_core_array();
	}

	BENCH_END(sort, "sort", _time.size(), sizeof(uint64_t), _time.size(), sizeof(uint64_t));

	pvcop::db::indexes minmax_indexes(2);
	auto minmax_core_indexes = minmax_indexes.to_core_array();

	if (_sorted_indexes) {
		minmax_core_indexes[0] = _sort[0];
		minmax_core_indexes[1] = _sort[_time.size() - 1];
	} else {
		minmax_core_indexes[0] = 0;
		minmax_core_indexes[1] = _time.size() - 1;
	}

	_minmax = _time.join(minmax_indexes);
	_last_params = SamplingParams(0, 0, _minmax, 0, 0);
}

void Inendi::PVRangeSubSampler::set_sampling_count(size_t sampling_count)
{
	_sampling_count = sampling_count;
	_reset = true;
}

template <typename T>
static pvcop::db::array
_minmax_subrange(const pvcop::db::array& minmax, double first_ratio, double last_ratio)
{
	pvcop::db::array rel_minmax(minmax.formatter()->name(), 2);
	rel_minmax.set_formatter(minmax.formatter());

	pvcop::core::array<T> core_rel_minmax = rel_minmax.to_core_array<T>();
	const pvcop::core::array<T>& core_minmax = minmax.to_core_array<T>();

	T abs_min = core_minmax[0];
	T abs_max = core_minmax[1];

	auto duration = abs_max - abs_min;

	if
		constexpr(std::is_same<T, boost::posix_time::ptime>::value)
		{
			auto duration_us = duration.total_microseconds();
			core_rel_minmax[0] =
			    abs_min + boost::posix_time::microseconds(first_ratio * duration_us);
			core_rel_minmax[1] =
			    abs_min + boost::posix_time::microseconds(last_ratio * duration_us);
		}
	else {
		core_rel_minmax[0] = abs_min + (duration * first_ratio);
		core_rel_minmax[1] = abs_min + (duration * last_ratio);
	}

	return rel_minmax;
}

pvcop::db::array Inendi::PVRangeSubSampler::minmax_subrange(double first_ratio, double last_ratio)
{
	typedef pvcop::db::array (*minmax_subrange_func_t)(const pvcop::db::array&, double, double);
	using func_map_t = std::unordered_map<std::string, minmax_subrange_func_t>;
	static const func_map_t func_map = []() {
		func_map_t map;
		map.insert({"number_float", &_minmax_subrange<float>});
		map.insert({"number_double", &_minmax_subrange<double>});
		map.insert({"number_uint32", &_minmax_subrange<uint32_t>});
		map.insert({"number_uint64", &_minmax_subrange<uint64_t>});
		map.insert({"datetime", &_minmax_subrange<uint32_t>});
		map.insert({"datetime_ms", &_minmax_subrange<uint64_t>});
		map.insert({"datetime_us", &_minmax_subrange<boost::posix_time::ptime>});
		return map;
	}();

	auto minmax_subrange_f = func_map.at(_minmax.formatter()->name());
	return minmax_subrange_f(_minmax, first_ratio, last_ratio);
}

void Inendi::PVRangeSubSampler::subsample(double first_ratio,
                                          double last_ratio,
                                          double min_ratio /*= 0*/,
                                          double max_ratio /*= 0*/)
{
	const value_type min = min_ratio * std::numeric_limits<value_type>::max();
	const value_type max = max_ratio * std::numeric_limits<value_type>::max();

	subsample(minmax_subrange(first_ratio, last_ratio), min, max);
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

	typedef void (PVRangeSubSampler::*compute_histogram_func_t)(size_t, size_t,
	                                                            const pvcop::db::array&);
	using func_map_t = std::unordered_map<std::string, compute_histogram_func_t>;
	static const func_map_t func_map = [&]() {
		func_map_t map;
		map.insert({"number_float", &PVRangeSubSampler::compute_histogram<float>});
		map.insert({"number_double", &PVRangeSubSampler::compute_histogram<double>});
		map.insert({"number_uint32", &PVRangeSubSampler::compute_histogram<uint32_t>});
		map.insert({"number_uint64", &PVRangeSubSampler::compute_histogram<uint64_t>});
		map.insert({"datetime", &PVRangeSubSampler::compute_histogram<uint32_t>});
		map.insert({"datetime_ms", &PVRangeSubSampler::compute_histogram<uint64_t>});
		map.insert(
		    {"datetime_us", &PVRangeSubSampler::compute_histogram<boost::posix_time::ptime>});
		return map;
	}();

	if (_reset) {
		allocate_internal_structures(); // start and end are not included
		_reset = false;
	}

	auto compute_ranges_values_count_f = func_map.at(_time.formatter()->name());
	((*this).*compute_ranges_values_count_f)(first, last, minmax);
	compute_ranges_average(first, last, min, max);

	_subsampled.emit();
}

void Inendi::PVRangeSubSampler::set_selected_timeseries(
    const std::unordered_set<size_t>& selected_timeseries /* = {} */)
{
	if (not selected_timeseries.empty()) {
		_timeseries_to_subsample.clear();
		std::copy_if(selected_timeseries.begin(), selected_timeseries.end(),
		             std::back_inserter(_timeseries_to_subsample), [this](size_t index) {
			             return _selected_timeseries.find(index) == _selected_timeseries.end();
			         });
		_selected_timeseries = selected_timeseries;
	}
}

void Inendi::PVRangeSubSampler::resubsample(const std::unordered_set<size_t>& timeseries /*= {}*/)
{
	if (not timeseries.empty()) {
		_timeseries_to_subsample.clear();
		std::copy(timeseries.begin(), timeseries.end(),
		          std::back_inserter(_timeseries_to_subsample));
	} else {
		_timeseries_to_subsample =
		    std::vector<size_t>(_selected_timeseries.begin(), _selected_timeseries.end());
	}
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

#pragma omp parallel for
	for (size_t k = 0; k < _timeseries_to_subsample.size(); k++) {

		const size_t i = _timeseries_to_subsample[k];

		size_t start = first;
		size_t end = first;
		const pvcop::core::array<value_type>& timeserie = _timeseries[i];
		for (size_t j = 0; j < _histogram.size(); j++) {
			const size_t values_count = _histogram[j];
			end += values_count;
			const size_t selected_values_count =
			    (start != end) ? pvcop::core::algo::bit_count(_sel, start, end - 1) : 0;
			uint64_t sum = 0;
			for (size_t k = start; k < end; k++) {
				auto v = not _sort ? k : _sort[k];
				if (_sel[v]) {
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
					_avg_matrix[i][j] = (display_type)(((double)(raw_value - min) / (max - min)) *
					                                   display_type_max_val); // nominal value
				}
			}
		}
	}

	BENCH_END(computing_average, "computing_average", _time.size(), sizeof(uint64_t),
	          _sampling_count, sizeof(uint64_t));
}
