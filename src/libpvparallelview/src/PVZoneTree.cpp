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
#include <squey/PVScaled.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCode.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVZoneProcessing.h>
#include <pvparallelview/PVZoneTree.h>

#include <boost/static_assert.hpp>

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_sort.h>
#include <tbb/task_group.h>

#include <omp.h>

#define GRAINSIZE 128

namespace
{
//! Reset a whole per-bucket buffer. Serial, this alone costs more than the
//! filtering that follows it.
void reset_elts(PVRow* buf_elts)
{
	tbb::parallel_for(tbb::blocked_range<size_t>(0, NBUCKETS, 65536),
	                  [buf_elts](tbb::blocked_range<size_t> const& r) {
		                  std::fill_n(buf_elts + r.begin(), r.size(), PVROW_INVALID_VALUE);
	                  });
}

//! Buckets are grouped into partitions, and a partition is the unit everything
//! after the counting works on: its buckets are consecutive, so its rows land in
//! one unbroken stretch of the row store.
//!
//! Few enough of them that a task writing one row per partition writes to a
//! handful of places the CPU can combine, and small enough that one partition's
//! stretch of the row store stays in cache while it is being sorted.
constexpr uint32_t PART_BITS = 6;
constexpr uint32_t NPARTS = 1u << PART_BITS;
constexpr uint32_t PART_SHIFT = 2 * NBITS_INDEX - PART_BITS;
constexpr size_t PART_BUCKETS = size_t(1) << PART_SHIFT;
constexpr uint32_t PART_MASK = (1u << PART_SHIFT) - 1;
static_assert(NPARTS * PART_BUCKETS == NBUCKETS, "partitions must cover the buckets exactly");

//! The bucket a pair of scaled values falls into.
inline uint32_t bucket_of(uint32_t y1, uint32_t y2)
{
	return (y1 >> (32 - NBITS_INDEX)) | ((y2 >> (32 - NBITS_INDEX)) << NBITS_INDEX);
}
} // namespace

using Squey::PVSelection;

// PVZoneTree implementation
//

PVParallelView::PVZoneTree::PVZoneTree() : PVZoneTreeBase()
{
}

/**
 * Sort every row into its bucket.
 *
 * Rows are placed in two steps. First each row goes to one of a handful of
 * partitions -- few enough destinations that the CPU can combine the writes --
 * and only then, one partition at a time, to its bucket within it. A partition's
 * buckets are consecutive, so its rows occupy one unbroken stretch of the row
 * store, small enough to stay in cache while it is sorted; scattering over the
 * whole store at once misses on nearly every row once it outgrows the cache.
 *
 * Nothing is ever counted per bucket per task. The first step only needs to know
 * how many rows go to each partition, which is a handful of counters per task;
 * how many go to each bucket is then counted inside a partition, from what it
 * has in hand. Holding a growable list per bucket per task instead -- a million
 * of them, tens of megabytes a task -- cost more to walk and to give back than
 * either pass over the rows costs to run.
 *
 * Rows keep the order they had: a task writes ahead of every task after it, and
 * within a task in increasing row order.
 */
void PVParallelView::PVZoneTree::process_tbb_sse_treeb(PVZoneProcessing const& zp,
                                                       ProcessData& pdata)
{
	const PVRow nrows = zp.size;
	const uint32_t* const pcol_a = zp.scaled_a;
	const uint32_t* const pcol_b = zp.scaled_b;

	// Rows are shared out in fixed-size ranges, with a floor on the range size so
	// that a small zone does not pay for more tasks than it can keep busy.
	const size_t step = std::max<size_t>((nrows + pdata.max_tasks() - 1) / pdata.max_tasks(),
	                                     TREE_CREATION_GRAINSIZE);
	const uint32_t ntasks =
	    nrows ? (uint32_t)std::min<size_t>(pdata.max_tasks(), (nrows + step - 1) / step) : 1;

	const auto task_range = [&](uint32_t t) {
		const PVRow begin = (PVRow)std::min<size_t>(nrows, (size_t)t * step);
		return std::pair<PVRow, PVRow>{begin, (PVRow)std::min<size_t>(nrows, begin + step)};
	};

	// How many rows each task sends to each partition. A few hundred counters in
	// all, so this pass keeps them in the innermost cache whatever the zone holds.
	std::vector<size_t> task_part_rows((size_t)ntasks * NPARTS, 0);

	BENCH_START(count);
	tbb::parallel_for(uint32_t(0), ntasks, [&](uint32_t t) {
		size_t counts[NPARTS] = {};
		const auto [begin, end] = task_range(t);
		for (PVRow r = begin; r < end; r++) {
			counts[bucket_of(pcol_a[r], pcol_b[r]) >> PART_SHIFT]++;
		}
		std::copy_n(counts, NPARTS, task_part_rows.begin() + (size_t)t * NPARTS);
	});
	BENCH_END(count, "COUNT", nrows * 2, sizeof(uint32_t), NPARTS, sizeof(size_t));

	// Where each partition starts, and where within it each task writes. Tasks in
	// order, so a partition comes out sorted by row.
	BENCH_START(offsets);
	std::vector<size_t> part_start(NPARTS + 1);
	std::vector<size_t> task_off((size_t)NPARTS * ntasks);
	size_t running = 0;
	for (uint32_t p = 0; p < NPARTS; p++) {
		part_start[p] = running;
		for (uint32_t t = 0; t < ntasks; t++) {
			task_off[(size_t)p * ntasks + t] = running;
			running += task_part_rows[(size_t)t * NPARTS + p];
		}
	}
	part_start[NPARTS] = running;
	assert(running == nrows);

	// Reuse the store when the zone has not changed size: at hundreds of millions
	// of rows, handing gigabytes back only to ask for them again costs more than
	// the counting pass above.
	if (_tree_data_size != nrows) {
		if (_tree_data) {
			PVCore::PVAlignedAllocator<PVRow, 4>().deallocate(_tree_data, 0);
		}
		_tree_data = PVCore::PVAlignedAllocator<PVRow, 16>().allocate(nrows);
		_tree_data_size = nrows;
	}
	BENCH_END(offsets, "OFFSETS", NPARTS, sizeof(size_t), NPARTS, sizeof(size_t));

	// First step: rows to their partition. The bucket's low bits ride along, so
	// the second step never has to come back to the scaled columns for them.
	BENCH_START(partition);
	std::vector<uint16_t> low_bits(nrows);
	tbb::parallel_for(uint32_t(0), ntasks, [&](uint32_t t) {
		size_t at[NPARTS];
		for (uint32_t p = 0; p < NPARTS; p++) {
			at[p] = task_off[(size_t)p * ntasks + t];
		}
		const auto [begin, end] = task_range(t);
		for (PVRow r = begin; r < end; r++) {
			const uint32_t b = bucket_of(pcol_a[r], pcol_b[r]);
			const size_t pos = at[b >> PART_SHIFT]++;
			_tree_data[pos] = r;
			low_bits[pos] = (uint16_t)(b & PART_MASK);
		}
	});
	BENCH_END(partition, "PARTITION", nrows * 2, sizeof(uint32_t), nrows, sizeof(PVRow));

	// Second step: within one partition, count what each of its buckets holds,
	// then place the rows. Both what is read and what is written now sit in one
	// stretch of the row store, so this stays in cache however many buckets the
	// zone spreads over.
	BENCH_START(scatter);
	std::vector<size_t> part_occupied(NPARTS);
	tbb::parallel_for(uint32_t(0), NPARTS, [&](uint32_t p) {
		const size_t first = part_start[p];
		const size_t n = part_start[p + 1] - first;
		const size_t base = (size_t)p * PART_BUCKETS;

		if (n == 0) {
			for (size_t i = 0; i < PART_BUCKETS; i++) {
				_treeb[base + i] = PVBranch{nullptr, 0};
				_first_elts[base + i] = PVROW_INVALID_VALUE;
			}
			part_occupied[p] = 0;
			return;
		}

		// Read out before the scatter overwrites this stretch.
		const std::vector<PVRow> rows(_tree_data + first, _tree_data + first + n);
		const std::vector<uint16_t> lows(low_bits.begin() + first, low_bits.begin() + first + n);

		std::vector<size_t> at(PART_BUCKETS, 0);
		for (size_t i = 0; i < n; i++) {
			at[lows[i]]++;
		}

		size_t cur = first;
		size_t occupied = 0;
		for (size_t i = 0; i < PART_BUCKETS; i++) {
			const size_t count = at[i];
			PVBranch& branch = _treeb[base + i];
			branch.count = count;
			branch.p = count ? (_tree_data + cur) : nullptr;
			if (count == 0) {
				_first_elts[base + i] = PVROW_INVALID_VALUE;
			} else {
				occupied++;
			}
			at[i] = cur;
			cur += count;
		}
		part_occupied[p] = occupied;

		for (size_t i = 0; i < n; i++) {
			_tree_data[at[lows[i]]++] = rows[i];
		}
	});
	BENCH_END(scatter, "SCATTER", nrows, sizeof(uint16_t), nrows, sizeof(PVRow));

	// List the buckets worth visiting later on, and the lowest row of each -- the
	// first one it was given, the ranges having been walked in order.
	size_t total_occupied = 0;
	for (uint32_t p = 0; p < NPARTS; p++) {
		const size_t occupied = part_occupied[p];
		part_occupied[p] = total_occupied;
		total_occupied += occupied;
	}
	_occupied_branches.resize(total_occupied);
	_sel_counts.assign(total_occupied, 0);
	_bg_counts.assign(total_occupied, 0);

	tbb::parallel_for(uint32_t(0), NPARTS, [&](uint32_t p) {
		const size_t base = (size_t)p * PART_BUCKETS;
		size_t occ_idx = part_occupied[p];
		for (size_t i = 0; i < PART_BUCKETS; i++) {
			PVBranch const& branch = _treeb[base + i];
			if (branch.count) {
				_occupied_branches[occ_idx++] = (uint32_t)(base + i);
				_first_elts[base + i] = branch.p[0];
			}
		}
	});
}

void PVParallelView::PVZoneTree::filter_by_sel_tbb_treeb(Squey::PVSelection const& sel,
                                                         PVRow* buf_elts,
                                                         uint32_t* counts)
{
	reset_elts(buf_elts);

	tbb::parallel_for(tbb::blocked_range<size_t>(0, _occupied_branches.size(), GRAINSIZE),
	                  [this, &sel, buf_elts, counts](tbb::blocked_range<size_t> const& br) {
		                  const auto selected = [&sel](PVRow v) { return sel.get_line_fast(v); };
		                  for (size_t i = br.begin(); i != br.end(); i++) {
			                  const uint32_t b = _occupied_branches[i];
			                  PVRow* end = _treeb[b].p + _treeb[b].count;
			                  PVRow* res = std::find_if(_treeb[b].p, end, selected);
			                  if (res != end) {
				                  buf_elts[b] = *res;
			                  }
			                  if (counts != nullptr) {
				                  counts[i] =
				                      res != end ? 1 + std::count_if(res + 1, end, selected) : 0;
			                  }
		                  }
		              },
	                  tbb::simple_partitioner());
}

void PVParallelView::PVZoneTree::filter_by_sel_background_tbb_treeb(Squey::PVSelection const& sel,
                                                                    PVRow* buf_elts,
                                                                    uint32_t* counts)
{
	// returns a zone tree with only the selected events
	Squey::PVSelection::const_pointer sel_buf = sel.get_buffer();
	if (sel_buf == nullptr) {
		// Empty selection
		memcpy(buf_elts, _first_elts, sizeof(PVRow) * NBUCKETS);
		if (counts != nullptr) {
			for (size_t i = 0; i < _occupied_branches.size(); i++) {
				counts[i] = _treeb[_occupied_branches[i]].count;
			}
		}
		return;
	}
	BENCH_START(subtree2);
	reset_elts(buf_elts);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, _occupied_branches.size(), GRAINSIZE),
	                  [&](const tbb::blocked_range<size_t>& range) {
		                  PVRow* buf_elts_ = buf_elts;
		                  PVZoneTree* tree = this;
		                  const auto selected = [sel_buf](PVRow r) {
			                  return (sel_buf[PVSelection::line_index_to_chunk(r)] &
			                          ((PVSelection::chunk_t)1
			                           << (PVSelection::line_index_to_chunk_bit(r)))) != 0;
		                  };
		                  for (size_t i = range.begin(); i != range.end(); i++) {
			                  const uint32_t b = tree->_occupied_branches[i];
			                  PVRow res = PVROW_INVALID_VALUE;
			                  if (tree->branch_valid(b)) {
				                  const PVRow r = tree->get_first_elt_of_branch(b);
				                  if (selected(r)) {
					                  res = r;
				                  } else {
					                  for (size_t i = 0; i < tree->_treeb[b].count; i++) {
						                  const PVRow r = tree->_treeb[b].p[i];
						                  if (selected(r)) {
							                  res = r;
							                  break;
						                  }
					                  }
				                  }
				                  // If nothing from the nu_selection, take the first event (a
				                  // zombie one)
				                  if (res == PVROW_INVALID_VALUE) {
					                  res = r;
				                  }
				                  if (counts != nullptr) {
					                  // A zombie stands for every row of its bucket.
					                  const PVRow* rows = tree->_treeb[b].p;
					                  const uint32_t count = tree->_treeb[b].count;
					                  const auto kept = std::count_if(rows, rows + count, selected);
					                  counts[i] = kept > 0 ? kept : count;
				                  }
			                  }
			                  buf_elts_[b] = res;
		                  }
		              });
	BENCH_END(subtree2, "filter_by_sel_background_tbb_treeb", 1, 1, sizeof(PVRow), NBUCKETS);
}

void PVParallelView::PVZoneTree::dump_branches() const
{
	for (size_t i = 0; i < NBUCKETS; i++) {
		if (branch_valid(i) > 0) {
			std::cout << "branch " << i << ": ";
			for (size_t r = 0; r < get_branch_count(i); r++) {
				std::cout << get_branch_element(i, r) << ",";
			}
			std::cout << std::endl;
		}
	}
}

bool PVParallelView::PVZoneTree::operator==(PVParallelView::PVZoneTree& zt) const
{
	for (size_t i = 0; i < NBUCKETS; ++i) {
		if (get_branch_count(i) != zt.get_branch_count(i)) {
			return false;
		}
		for (size_t r = 0; r < get_branch_count(i); ++r) {
			if (get_branch_element(i, r) != zt.get_branch_element(i, r)) {
				return false;
			}
		}
	}

	if (memcmp(_first_elts, zt._first_elts, NBUCKETS * sizeof(PVRow)) != 0) {
		return false;
	} else if (memcmp(_sel_elts, zt._sel_elts, NBUCKETS * sizeof(PVRow)) != 0) {
		return false;
	} else if (memcmp(_bg_elts, zt._bg_elts, NBUCKETS * sizeof(PVRow)) != 0) {
		return false;
	}

	return true;
}
