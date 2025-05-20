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

#include "PVMappingFilterMacAddress.h"

#include <pvkernel/core/squey_bench.h>
#include <pvkernel/rush/PVNraw.h>

#include <squey/PVSelection.h>

#include <tbb/parallel_sort.h>

#include <unordered_map>
#include <limits>

/*****************************************************************************
 *
 *****************************************************************************/

static constexpr const size_t vendor_max_count = 1 << 24;
static constexpr const size_t nic_max_count = 1 << 24;

static constexpr const size_t invalid_vendor = 1 << 24;

static constexpr const size_t mapping_range = (size_t)1 << 32;

using mac_t = uint64_t;
using vendor_t = uint64_t;
using nic_t = uint64_t;

inline vendor_t mac_to_vendor(mac_t mac)
{
	return mac >> 24;
}

inline nic_t mac_to_nic(mac_t mac)
{
	return mac % nic_max_count;
}

/*****************************************************************************
 *
 *****************************************************************************/

using sizes_t = std::vector<size_t>;

/**
 * This function computes informations needed to do vendor based uniform mapping.
 *
 * @param data_array the column data.
 * @param vendor_bases the vendors base value (in mapping space)
 *
 * @return the number of found vendors
 */
static size_t compute_uniform_vendor_bases(const pvcop::db::array& data_array,
                                           sizes_t& vendor_bases)
{
	const auto& data = data_array.to_core_array<uint64_t>();

	Squey::PVSelection vendor_bits(vendor_max_count);
	vendor_bits.select_none();

	/* first, we construct the "bitfield" of existing vendors
	 */
	for (const auto value : data) {
		vendor_bits.set_bit_fast(mac_to_vendor(value));
	}

	sizes_t bases(vendor_max_count, 0);
	size_t vendor_count = 0;

	/* we can also compute their index in the unique vendor space
	 */
	vendor_bits.visit_selected_lines_serial([&](size_t i) {
		bases[i] = vendor_count;
		++vendor_count;
	});

/* finally, we compute their position in the mapping space.
 */
#pragma omp parallel for
	for (size_t i = 0; i < vendor_max_count; ++i) {
		bases[i] = (bases[i] * mapping_range) / vendor_count;
	}

	vendor_bases = std::move(bases);

	return vendor_count;
}

/* This function computes informations to do NIC based uniform mapping.
 *
 * @param data_array the column data.
 * @param vendor_counts the vendors unique NIC counts
 * @param mac_indexes the MAC indexes relative to their vendor part.
 */
static void compute_mac_distribution(const pvcop::db::array& data_array,
                                     sizes_t& vendor_counts,
                                     sizes_t& mac_indexes)
{
	const auto& data = data_array.to_core_array<uint64_t>();
	pvcop::db::indexes idexes_array(data_array.size());

	sizes_t counts(vendor_max_count, 0);
	sizes_t indexes(data_array.size(), 0);

	/**
	 * the goal is to compute MAC addresses distribution from the distinct values
	 * set and not from the whole column:
	 * - the per-vendor count is deduced from the histogram;
	 * - the vendor relative MAC index is deduced from the sorted distinct MAC list.
	 */
	pvcop::db::array uniq_array;

	/* fistst, the distinct.
	 */
	pvcop::db::algo::distinct(data_array, uniq_array);

	auto& uniq = uniq_array.to_core_array<uint64_t>();

	/* we compute per-vendor unique MAC count
	 */
	for (uint64_t i : uniq) {
		++counts[mac_to_vendor(i)];
	}

	/* now, we'll continue using the sorted distinct MAC list
	 */
	tbb::parallel_sort(uniq.begin(), uniq.end());

	/* we compute the MAC address index relatively to its vendor
	 */
	size_t count = 0;
	vendor_t prev_vendor = invalid_vendor;

	std::unordered_map<size_t, size_t> relative_indices(uniq.size());

	for (const auto mac : uniq) {
		const vendor_t vendor = mac_to_vendor(mac);

		if (vendor != prev_vendor) {
			// new vendor (and implictly new MAC), we restart the counter.
			prev_vendor = vendor;
			count = 0;
		} else {
			// new MAC (but same vendor)
			++count;
		}

		relative_indices.emplace(mac, count);
	}

/* finally, we can set the vendor-relative index for each MAC address
 */
#pragma omp parallel for
	for (size_t i = 0; i < data.size(); ++i) {
		indexes[i] = relative_indices.at(data[i]);
	}

	vendor_counts = std::move(counts);
	mac_indexes = std::move(indexes);
}

/*****************************************************************************
 *
 * Squey::PVMappingFilterMacAddressL::operator()
 *
 *****************************************************************************/

pvcop::db::array Squey::PVMappingFilterMacAddressL::operator()(PVCol const col,
                                                                PVRush::PVNraw const& nraw)
{
	const pvcop::db::array& data_array = nraw.column(col);
	const auto& data = data_array.to_core_array<uint64_t>();

	pvcop::db::array mapping_array("number_uint32", data_array.size());
	auto& mapping = mapping_array.to_core_array<uint32_t>();

	BENCH_START(whole);

	/* built an uint32 from the 32 most significant bits of a 48 bits value
	 */
	std::transform(data.begin(), data.end(), mapping.begin(),
	               [](const uint64_t v) { return v >> 16; });

	BENCH_END(whole, "PVMappingFilterMacAddressL::operator()", data_array.size(), sizeof(uint64_t),
	          data_array.size(), sizeof(uint32_t));

	return mapping_array;
}

/*****************************************************************************
 *
 * Squey::PVMappingFilterMacAddressLU::operator()
 *
 *****************************************************************************/

pvcop::db::array Squey::PVMappingFilterMacAddressLU::operator()(PVCol const col,
                                                                 PVRush::PVNraw const& nraw)
{
	const pvcop::db::array& data_array = nraw.column(col);
	const auto& data = data_array.to_core_array<uint64_t>();

	pvcop::db::array mapping_array("number_uint32", data_array.size());
	auto& mapping = mapping_array.to_core_array<uint32_t>();

	BENCH_START(whole);

	/* compute information for uniform MACs distribution
	 */
	sizes_t vendor_counts;
	sizes_t mac_indexes;

	compute_mac_distribution(data_array, vendor_counts, mac_indexes);

/* As vendor are linearly distributed (on 24 bits), NIC will be relatively distributed in
 * the range [0;256) relatively to the vendor offset.
 */
#pragma omp parallel for
	for (size_t i = 0; i < data.size(); ++i) {
		const vendor_t vendor = mac_to_vendor(data[i]);
		const size_t vendor_base = vendor * 256;
		const size_t nic_offset = (mac_indexes[i] * 256) / vendor_counts[vendor];

		mapping[i] = vendor_base + nic_offset;
	}

	BENCH_END(whole, "PVMappingFilterMacAddressLU::operator()", data_array.size(), sizeof(uint64_t),
	          data_array.size(), sizeof(uint32_t));

	return mapping_array;
}

/*****************************************************************************
 *
 * Squey::PVMappingFilterMacAddressUL::operator()
 *
 *****************************************************************************/

pvcop::db::array Squey::PVMappingFilterMacAddressUL::operator()(PVCol const col,
                                                                 PVRush::PVNraw const& nraw)
{
	const pvcop::db::array& data_array = nraw.column(col);
	const auto& data = data_array.to_core_array<uint64_t>();

	pvcop::db::array mapping_array("number_uint32", data_array.size());
	auto& mapping = mapping_array.to_core_array<uint32_t>();

	BENCH_START(whole);

	/* compute informations to distribute uniformly vendors
	 */
	sizes_t vendor_bases;
	const size_t vendor_count = compute_uniform_vendor_bases(data_array, vendor_bases);

/* As the N vendors are uniformly distributed, NIC will be distributed in
 * the range [0;2^32/N) relatively to the vendor offset.
 */
#pragma omp parallel for
	for (size_t i = 0; i < data.size(); ++i) {
		const size_t vendor_base = vendor_bases[mac_to_vendor(data[i])];
		const size_t nic_offset =
		    (mac_to_nic(data[i]) * mapping_range) / (vendor_count * nic_max_count);

		mapping[i] = vendor_base + nic_offset;
	}

	BENCH_END(whole, "PVMappingFilterMacAddressUL::operator()", data_array.size(), sizeof(uint64_t),
	          data_array.size(), sizeof(uint32_t));

	return mapping_array;
}

/*****************************************************************************
 *
 * Squey::PVMappingFilterMacAddressUU::operator()
 *
 *****************************************************************************/

pvcop::db::array Squey::PVMappingFilterMacAddressUU::operator()(PVCol const col,
                                                                 PVRush::PVNraw const& nraw)
{
	const pvcop::db::array& data_array = nraw.column(col);
	const auto& data = data_array.to_core_array<uint64_t>();

	pvcop::db::array mapping_array("number_uint32", data_array.size());
	auto& mapping = mapping_array.to_core_array<uint32_t>();

	BENCH_START(whole);

	/* compute informations to distribute uniformly vendors
	 */
	sizes_t vendor_bases;
	const size_t vendor_count = compute_uniform_vendor_bases(data_array, vendor_bases);

	/* compute informations to distribute uniformly MAC
	 */
	sizes_t vendor_counts;
	sizes_t mac_indexes;

	compute_mac_distribution(data_array, vendor_counts, mac_indexes);

/* As the N vendors are uniformly distributed, NIC will be distributed in
 * the range [0;2^32/N) relatively to the vendor offset.
 */
#pragma omp parallel for
	for (size_t i = 0; i < data.size(); ++i) {
		const vendor_t vendor = mac_to_vendor(data[i]);
		const size_t vendor_base = vendor_bases[vendor];
		const size_t nic_offset =
		    (mac_indexes[i] * mapping_range) / (vendor_count * vendor_counts[vendor]);

		mapping[i] = vendor_base + nic_offset;
	}

	BENCH_END(whole, "PVMappingFilterMacAddressUU::operator()", data_array.size(), sizeof(uint64_t),
	          data_array.size(), sizeof(uint32_t));

	return mapping_array;
}
