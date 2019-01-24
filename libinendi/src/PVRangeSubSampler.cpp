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
    size_t sampling_count /*= 2048*/)
    : _time(time), _timeseries(timeseries), _minmax()
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
}

void Inendi::PVRangeSubSampler::set_sampling_count(size_t sampling_count)
{
	_sampling_count = sampling_count;

	allocate_internal_structures(); // start and end are not included
}

template <typename T>
static pvcop::db::array
minmax_subrange(const pvcop::db::array& minmax, double first_ratio, double last_ratio)
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

void Inendi::PVRangeSubSampler::subsample(double first_ratio,
                                          double last_ratio,
                                          double min_ratio /*= 0*/,
                                          double max_ratio /*= 0*/)
{
	typedef pvcop::db::array (*minmax_subrange_func_t)(const pvcop::db::array&, double, double);
	using func_map_t = std::unordered_map<std::string, minmax_subrange_func_t>;
	static const func_map_t func_map = [&]() {
		func_map_t map;
		map.insert({"number_uint32", &minmax_subrange<uint32_t>});
		map.insert({"number_uint64", &minmax_subrange<uint64_t>});
		map.insert({"datetime", &minmax_subrange<uint32_t>});
		map.insert({"datetime_ms", &minmax_subrange<uint64_t>});
		map.insert({"datetime_us", &minmax_subrange<boost::posix_time::ptime>});
		return map;
	}();

	const value_type min = min_ratio * std::numeric_limits<value_type>::max();
	const value_type max = max_ratio * std::numeric_limits<value_type>::max();

	auto minmax_subrange_f = func_map.at(_minmax.formatter()->name());
	subsample(minmax_subrange_f(_minmax, first_ratio, last_ratio), min, max);
}

void Inendi::PVRangeSubSampler::subsample(const pvcop::db::array& minmax,
                                          size_t min /*= 0*/,
                                          size_t max /*= 0*/)
{
	auto[first, last] = _time.equal_range(minmax, _sorted_indexes);
	subsample(first, last, min, max);
}

void Inendi::PVRangeSubSampler::subsample(size_t first /*= 0*/,
                                          size_t last /*= 0*/,
                                          size_t min /*= 0*/,
                                          size_t max /*= 0*/)
{
	if (last == 0) {
		last = _time.size();
	}
	if (max == 0) {
		max = std::numeric_limits<size_t>::max();
	}
	_first = first;
	_last = last;
	_min = min;
	_max = max;

	typedef void (PVRangeSubSampler::*compute_ranges_values_count_func_t)(size_t, size_t);
	using func_map_t = std::unordered_map<std::string, compute_ranges_values_count_func_t>;
	static const func_map_t func_map = [&]() {
		func_map_t map;
		map.insert({"number_uint32", &PVRangeSubSampler::compute_ranges_values_count<uint32_t>});
		map.insert({"number_uint64", &PVRangeSubSampler::compute_ranges_values_count<uint64_t>});
		map.insert({"datetime", &PVRangeSubSampler::compute_ranges_values_count<uint32_t>});
		map.insert({"datetime_ms", &PVRangeSubSampler::compute_ranges_values_count<uint64_t>});
		map.insert({"datetime_us",
		            &PVRangeSubSampler::compute_ranges_values_count<boost::posix_time::ptime>});
		return map;
	}();

	auto compute_ranges_values_count_f = func_map.at(_time.formatter()->name());
	((*this).*compute_ranges_values_count_f)(first, last);
	compute_ranges_average(first, last, min, max);
}

void Inendi::PVRangeSubSampler::resubsample()
{
	subsample(_first, _last, _min, _max);
}

void Inendi::PVRangeSubSampler::allocate_internal_structures()
{
	// time interval array
	_sampled_time = pvcop::db::array(_time.formatter()->name(), _sampling_count - 2);
	_sampled_time.set_formatter(_time.formatter());

	_time_iota = std::vector<display_type>(_sampled_time.size());
	std::iota(_time_iota.begin(), _time_iota.end(), 0);

	// range values count
	_ranges_values_counts = std::vector<size_t>(_sampling_count);

	// matrix of average values
	_avg_matrix.resize(_timeseries.size() /* row count*/,
	                   std::vector<display_type>(_ranges_values_counts.size()));
}

void Inendi::PVRangeSubSampler::compute_ranges_average(size_t first,
                                                       size_t /*last*/,
                                                       size_t min,
                                                       size_t max)
{
	BENCH_START(computing_average);

#pragma omp parallel for
	for (size_t i = 0; i < _timeseries.size(); i++) {
		size_t start = first;
		size_t end = first;
		const pvcop::core::array<value_type>& timeserie = _timeseries[i];
		for (size_t j = 0; j < _ranges_values_counts.size(); j++) {
			const size_t value_count = _ranges_values_counts[j];
			end += value_count;
			uint64_t sum = 0;
			for (size_t k = start; k < end; k++) {
				sum += (std::numeric_limits<value_type>::max() -
				        (_sort ? timeserie[_sort[k]] : timeserie[k]));
			}
			start = end;
			if (value_count == 0) {
				_avg_matrix[i][j] = no_value; // no value in range
			} else {
				const uint64_t raw_value = sum / value_count;
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
	          _sampled_time.size(), sizeof(uint64_t));
}
