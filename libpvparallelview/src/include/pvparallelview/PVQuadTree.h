/**
 * \file PVQuadTree.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PARALLELVIEW_PVQUADTREE_H
#define PARALLELVIEW_PVQUADTREE_H

#include <pvbase/types.h>

#include <bithacks.h>

#include <pvkernel/core/PVAlgorithms.h>
#include <pvkernel/core/PVHSVColor.h>
#include <pvkernel/core/PVAllocators.h>
#include <pvkernel/core/PVVector.h>

#include <picviz/PVSelection.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCICode.h>

/* TODO: try to move code into .cpp, etc.
 */

namespace PVParallelView {

#define SW 0
#define SE 1
#define NW 2
#define NE 3

#pragma pack(push)
#pragma pack(4)

struct PVQuadTreeEntry {
	uint32_t y1;
	uint32_t y2;
	PVRow    idx;

	PVQuadTreeEntry()
	{
	}

	PVQuadTreeEntry(uint32_t y1_, uint32_t y2_, PVRow r)
	{
		y1 = y1_;
		y2 = y2_;
		idx = r;
	}
};
#pragma pack(pop)


#define __PV_IMPL_QUADTREE_BUFFER_ENTRY_COUNT (4096 * 2048)
// we store bits
#define QUADTREE_BUFFER_SIZE (__PV_IMPL_QUADTREE_BUFFER_ENTRY_COUNT >> 5)

typedef uint32_t pv_quadtree_buffer_entry_t;

// typedef PVCore::PVVector<PVQuadTreeEntry, tbb::scalable_allocator<PVQuadTreeEntry> > pvquadtree_entries_t;
// typedef PVCore::PVVector<PVQuadTreeEntry, 1000, PVCore::PVJEMallocAllocator<PVQuadTreeEntry> > pvquadtree_entries_t;
typedef PVCore::PVVector<PVQuadTreeEntry> pvquadtree_entries_t;

template<int MAX_ELEMENTS_PER_NODE = 10000, int REALLOC_ELEMENT_COUNT = 1000, int PREALLOC_ELEMENT_COUNT = 0, size_t Bbits = NBITS_INDEX>
class PVQuadTree
{
	constexpr static uint32_t mask_int_ycoord = (((uint32_t)1)<<Bbits)-1;

	typedef std::function<bool(const PVQuadTreeEntry &entry)> test_entry_f;

public:
	PVQuadTree(uint32_t y1_min_value, uint32_t y1_max_value, uint32_t y2_min_value, uint32_t y2_max_value, int max_level)
	{
		uint32_t y1_mid = y1_min_value + ((y1_max_value - y1_min_value) >> 1);
		uint32_t y2_mid = y2_min_value + ((y2_max_value - y2_min_value) >> 1);

		init(y1_min_value, y1_mid, y2_min_value, y2_mid, max_level);
	}

	// CTOR to use with call to init()
	PVQuadTree()
	{
	}

	~PVQuadTree() {
		if (_nodes == 0) {
			_datas.clear();
		} else {
			delete [] _nodes;
		}
	}

	void init(uint32_t y1_min_value, uint32_t y1_mid_value, uint32_t y2_min_value, uint32_t y2_mid_value, int max_level)
	{
		_y1_min_value = y1_min_value;
		_y1_mid_value = y1_mid_value;
		_y2_min_value = y2_min_value;
		_y2_mid_value = y2_mid_value;
		_max_level = max_level;
		if (PREALLOC_ELEMENT_COUNT != 0) {
			_datas.reserve(PREALLOC_ELEMENT_COUNT);
		} else {
			_datas = pvquadtree_entries_t();
		}
		_nodes = 0;
	}

	void insert(const PVQuadTreeEntry &e) {
		// searching for the right child
		register PVQuadTree *qt = this;
		while (qt->_nodes != 0) {
			qt = &qt->_nodes[qt->compute_index(e)];
		}

		// insertion
		qt->_datas.push_back(e);

		// does the current node must be splitted?
		if ((qt->_datas.size() >= MAX_ELEMENTS_PER_NODE) && qt->_max_level) {
			qt->create_next_level();
		}
	}

	__attribute__((noinline)) void compact()
	{
		if (_nodes) {
			for (int i = 0; i < 4; ++i) {
				_nodes[i].compact();
			}
		} else {
			_datas.compact();
		}
	}

	inline size_t memory() const
	{
		size_t mem = sizeof (PVQuadTree) - sizeof(pvquadtree_entries_t) + _datas.memory();
		if(_nodes != 0) {
			mem += _nodes[0].memory();
			mem += _nodes[1].memory();
			mem += _nodes[2].memory();
                        mem += _nodes[3].memory();
		}
		return mem;
	}

	inline size_t get_first_from_y1(uint64_t y1_min, uint64_t y1_max, uint32_t zoom,
	                                uint32_t y2_count,
	                                PVQuadTreeEntry *result,
	                                pv_quadtree_buffer_entry_t *buffer) const
	{
		return visit_y1::get_n_m(*this, y1_min, y1_max, zoom, y2_count,
		                         [&](const PVQuadTreeEntry &e) -> bool
		                         {
			                         return (e.y1 >= y1_min) && (e.y1 < y1_max);
		                         },
		                         result, buffer);
	}


	inline size_t get_first_from_y2(uint64_t y2_min, uint64_t y2_max, uint32_t zoom,
	                                uint32_t y1_count,
	                                PVQuadTreeEntry *result,
	                                pv_quadtree_buffer_entry_t *buffer) const
	{
		return visit_y2::get_n_m(*this, y2_min, y2_max, zoom, y1_count,
		                         [&](const PVQuadTreeEntry &e) -> bool
		                         {
			                         return (e.y2 >= y2_min) && (e.y2 < y2_max);
		                         },
		                         result, buffer);
	}


	inline size_t get_first_sel_from_y1(uint64_t y1_min, uint64_t y1_max,
	                                    const Picviz::PVSelection &selection,
	                                    uint32_t zoom, uint32_t y2_count,
	                                    PVQuadTreeEntry *result,
	                                    pv_quadtree_buffer_entry_t *buffer) const
	{
		return visit_y1::get_n_m(*this, y1_min, y1_max, zoom, y2_count,
		                         [&](const PVQuadTreeEntry &e) -> bool
		                         {
			                         return (e.y1 >= y1_min) && (e.y1 < y1_max)
				                         && selection.get_line(e.idx);
		                         },
		                         result, buffer);
	}


	inline size_t get_first_sel_from_y2(uint64_t y2_min, uint64_t y2_max,
	                                    const Picviz::PVSelection &selection,
	                                    uint32_t zoom, uint32_t y1_count,
	                                    PVQuadTreeEntry *result,
	                                    pv_quadtree_buffer_entry_t *buffer) const
	{
		return visit_y2::get_n_m(*this, y2_min, y2_max, zoom, y1_count,
		                         [&](const PVQuadTreeEntry &e) -> bool
		                         {
			                         return (e.y2 >= y2_min) && (e.y2 < y2_max)
				                         && selection.get_line(e.idx);
		                         },
		                         result, buffer);
	}

	PVQuadTree *get_subtree_from_y1(uint32_t y1_min, uint32_t y1_max)
	{
		PVQuadTree *new_tree = new PVQuadTree(*this);
		new_tree->init(*this);
		get_subtree_from_y1(*new_tree, y1_min, y1_max);
		return new_tree;
	}

	PVQuadTree *get_subtree_from_y2(uint32_t y2_min, uint32_t y2_max)
	{
		PVQuadTree *new_tree = new PVQuadTree(*this);
		new_tree->init(*this);
		get_subtree_from_y2(*new_tree, y2_min, y2_max);
		return new_tree;
	}

	PVQuadTree *get_subtree_from_y1y2(uint32_t y1_min, uint32_t y1_max, uint32_t y2_min, uint32_t y2_max)
	{
		PVQuadTree *new_tree = new PVQuadTree(*this);
		new_tree->init(*this);
		get_subtree_from_y1y2(*new_tree, y1_min, y1_max, y2_min, y2_max);
		return new_tree;
	}

	PVQuadTree *get_subtree_from_selection(const Picviz::PVSelection &selection)
	{
		PVQuadTree *new_tree = new PVQuadTree(*this);
		new_tree->init(*this);
		get_subtree_from_selection(*new_tree, selection);
		return new_tree;
	}

	size_t compute_selection_y1(const uint64_t y1_min, const uint64_t y1_max, Picviz::PVSelection &selection) const
	{
		return compute_selection_y1(*this, y1_min, y1_max, selection);
	}

	size_t compute_selection_y2(const uint64_t y2_min, const uint64_t y2_max, Picviz::PVSelection &selection) const
	{
		return compute_selection_y2(*this, y2_min, y2_max, selection);
	}

private:
	void init(const PVQuadTree &qt)
	{
		_y1_min_value = qt._y1_min_value;
		_y1_mid_value = qt._y1_mid_value;
		_y2_min_value = qt._y2_min_value;
		_y2_mid_value = qt._y2_mid_value;
		_max_level = qt._max_level;
		_nodes = 0;
	}

	inline int compute_index(const PVQuadTreeEntry &e) const
	{
		return ((e.y2 > _y2_mid_value) << 1) | (e.y1 > _y1_mid_value);
	}

	void create_next_level()
	{
		uint32_t y1_step = (_y1_mid_value - _y1_min_value) >> 1;
		uint32_t y2_step = (_y2_mid_value - _y2_min_value) >> 1;

		_nodes = new PVQuadTree [4];
		_nodes[NE].init(_y1_mid_value, _y1_mid_value + y1_step,
		                _y2_mid_value, _y2_mid_value + y2_step,
		                _max_level - 1);

		_nodes[SE].init(_y1_mid_value, _y1_mid_value + y1_step,
		                _y2_min_value, _y2_min_value + y2_step,
		                _max_level - 1);

		_nodes[SW].init(_y1_min_value, _y1_min_value + y1_step,
		                _y2_min_value, _y2_min_value + y2_step,
		                _max_level - 1);

		_nodes[NW].init(_y1_min_value, _y1_min_value + y1_step,
		                _y2_mid_value, _y2_mid_value + y2_step,
		                _max_level - 1);

		for (unsigned i = 0; i < _datas.size(); ++i) {
			const PVQuadTreeEntry &e = _datas.at(i);
			_nodes[compute_index(e)]._datas.push_back(e);
		}
		_datas.clear();
	}

private:

	struct visit_y1
	{
		static size_t get_n_m(PVQuadTree const& obj,
		                      const uint64_t y1_min, const uint64_t y1_max,
		                      const uint32_t zoom, const uint32_t y2_count,
		                      const test_entry_f &test_f, PVQuadTreeEntry *result,
		                      pv_quadtree_buffer_entry_t *buffer)
		{
			size_t num = 0;
			if (zoom == 0) {
				/* need a 1 entry width band along y1
				 */
				num = get_1_m(obj, y1_min, y1_max, y2_count, test_f, result, buffer);
			} else if (y2_count == 1) {
				/* need a 1 entry width band along y2
				 */
				num = get_n_1(obj, y1_min, y1_max, zoom, test_f, result, buffer);
			} else if (obj._nodes != 0) {
				/* recursive search can be processed
				 */
				if (obj._y1_mid_value < y1_max) {
					num += get_n_m(obj._nodes[NE], y1_min, y1_max,
					               zoom - 1, y2_count >> 1, test_f,
					               result + num, buffer);
					num += get_n_m(obj._nodes[SE], y1_min, y1_max,
					               zoom - 1, y2_count >> 1, test_f,
					               result + num, buffer);
				}
				if (y1_min < obj._y1_mid_value) {
					num += get_n_m(obj._nodes[NW], y1_min, y1_max,
					               zoom - 1, y2_count >> 1, test_f,
					               result + num, buffer);
					num += get_n_m(obj._nodes[SW], y1_min, y1_max,
					               zoom - 1, y2_count >> 1, test_f,
					               result + num, buffer);
				}
			} else if (obj._datas.size() != 0) {
				/* this is a unsplitted node with data and an array of nxm
				 * entries is needed
				 */
				num = extract(obj, y1_min, y1_max, zoom, y2_count,
				              test_f, result, buffer);
			}
			return num;
		}

		static size_t get_1_m(PVQuadTree const& obj,
		                      const uint64_t y1_min, const uint64_t y1_max,
		                      const uint32_t y2_count,
		                      const test_entry_f &test_f, PVQuadTreeEntry *result,
		                      pv_quadtree_buffer_entry_t *buffer)
		{
			size_t num = 0;
			if (y2_count == 1) {
				/* time to extract
				 */
				result->idx = UINT32_MAX;
				if (get_1_1(obj, y1_min, y1_max, test_f, result)) {
					num = 1;
				}
			} else  if (obj._nodes != 0) {
				if (obj._y1_mid_value < y1_max) {
					num += get_1_m(obj._nodes[NE], y1_min, y1_max,
					               y2_count >> 1, test_f,
					               result + num, buffer);
					num += get_1_m(obj._nodes[SE], y1_min, y1_max,
					               y2_count >> 1, test_f,
					               result + num, buffer);
				}
				if (y1_min < obj._y1_mid_value) {
					num += get_1_m(obj._nodes[NW], y1_min, y1_max,
					               y2_count >> 1, test_f,
					               result + num, buffer);
					num += get_1_m(obj._nodes[SW], y1_min, y1_max,
					               y2_count >> 1, test_f,
					               result + num, buffer);
				}
			} else if (obj._datas.size() != 0) {
				/* this is a unsplitted node with data and an array of 1xm
				 * entries is needed
				 */
				num = extract(obj, y1_min, y1_max, 0, y2_count,
				              test_f, result, buffer);
			}
			return num;
		}

		static size_t get_n_1(PVQuadTree const& obj,
		                      const uint64_t y1_min, const uint64_t y1_max,
		                      const uint32_t zoom,
		                      const test_entry_f &test_f, PVQuadTreeEntry *result,
		                      pv_quadtree_buffer_entry_t *buffer)
		{
			size_t num = 0;
			if (zoom == 0) {
				/* time to extract
				 */
				result->idx = UINT32_MAX;
				if (get_1_1(obj, y1_min, y1_max, test_f, result)) {
					num = 1;
				}
			} else if (obj._nodes != 0) {
				if (obj._y1_mid_value < y1_max) {
					num += get_n_1(obj._nodes[NE], y1_min, y1_max,
					               zoom - 1, test_f,
					               result + num, buffer);
					num += get_n_1(obj._nodes[SE], y1_min, y1_max,
					               zoom - 1, test_f,
					               result + num, buffer);
				}
				if (y1_min < obj._y1_mid_value) {
					num += get_n_1(obj._nodes[NW], y1_min, y1_max,
					               zoom - 1, test_f,
					               result + num, buffer);
					num += get_n_1(obj._nodes[SW], y1_min, y1_max,
					               zoom - 1, test_f,
					               result + num, buffer);
				}
			} else if (obj._datas.size() != 0) {
				/* this is a unsplitted node with data and an array of nx1
				 * entries is needed
				 */
				num = extract(obj, y1_min, y1_max, zoom, 1,
				              test_f, result, buffer);
			}
			return num;
		}

		static bool get_1_1(PVQuadTree const& obj,
		                    const uint64_t y1_min, const uint64_t y1_max,
		                    const test_entry_f &test_f, PVQuadTreeEntry *result)
		{
			bool ret = false;
			if (obj._nodes != 0) {
				/* pick the first relevant element from the children
				 */
				if (obj._y1_mid_value < y1_max) {
					ret |= get_1_1(obj._nodes[NE], y1_min, y1_max, test_f, result);
					ret |= get_1_1(obj._nodes[SE], y1_min, y1_max, test_f, result);
				}
				if (y1_min < obj._y1_mid_value) {
					ret |= get_1_1(obj._nodes[NW], y1_min, y1_max, test_f, result);
					ret |= get_1_1(obj._nodes[SW], y1_min, y1_max, test_f, result);
				}
			} else {
				/* pick the first relevant element from the entry list
				 */
				for (size_t i = 0; i < obj._datas.size(); ++i) {
					const PVQuadTreeEntry &e = obj._datas.at(i);
					if (test_f(e) && (e.idx < result->idx)) {
						*result = e;
						return true;
					}
				}
			}
			return ret;
		}

		static size_t extract(PVQuadTree const& obj,
		                      const uint64_t y1_min, const uint64_t y1_max,
		                      const uint32_t zoom, const uint32_t y2_count,
		                      const test_entry_f &test_f, PVQuadTreeEntry *result,
		                      pv_quadtree_buffer_entry_t *buffer)
		{
			size_t num = 0;
			const uint64_t max_count = 1 << zoom;
			const uint64_t y1_orig = obj._y1_min_value;
			const uint64_t y1_len = (obj._y1_mid_value - y1_orig) * 2;
			const uint64_t y1_scale = y1_len / max_count;
			const uint64_t y2_orig = obj._y2_min_value;
			const uint64_t y2_scale = ((obj._y2_mid_value - y2_orig) * 2) / y2_count;
			const uint64_t ly1_min = (PVCore::clamp(y1_min, y1_orig, y1_orig + y1_len) - y1_orig) / y1_scale;
			const uint64_t ly1_max = (PVCore::clamp(y1_max, y1_orig, y1_orig + y1_len) - y1_orig) / y1_scale;
			const uint64_t clipped_max_count = PVCore::max(1UL, ly1_max - ly1_min);
			const size_t count_aligned = ((clipped_max_count * y2_count) + 31) / 32;
			memset(buffer, 0, count_aligned * sizeof(uint32_t));
			uint32_t remaining = clipped_max_count * y2_count;
			for(size_t i = 0; i < obj._datas.size(); ++i) {
				const PVQuadTreeEntry &e = obj._datas.at(i);
				if (!test_f(e)) {
					continue;
				}
				const uint32_t pos = (((e.y1 - y1_orig) / y1_scale) - ly1_min) + clipped_max_count * ((e.y2 - y2_orig) / y2_scale);
				if (B_IS_SET(buffer[pos >> 5], pos & 31)) {
					continue;
				}
				result[num] = e;
				B_SET(buffer[pos >> 5], pos & 31);
				++num;
				--remaining;
				if (remaining == 0) {
					break;
				}
			}
			return num;
		}
	};

	struct visit_y2
	{
		static size_t get_n_m(PVQuadTree const& obj,
		                      const uint64_t y2_min, const uint64_t y2_max,
		                      const uint32_t zoom, const uint32_t y1_count,
		                      const test_entry_f &test_f, PVQuadTreeEntry *result,
		                      pv_quadtree_buffer_entry_t *buffer)
		{
			size_t num = 0;
			if (zoom == 0) {
				/* need a 1 entry width band along y2
				 */
				num = get_1_m(obj, y2_min, y2_max, y1_count, test_f, result, buffer);
			} else if (y1_count == 1) {
				/* need a 1 entry width band along y1
				 */
				num = get_n_1(obj, y2_min, y2_max, zoom, test_f, result, buffer);
			} else if (obj._nodes != 0) {
				/* recursive search can be processed
				 */
				if (obj._y2_mid_value < y2_max) {
					num += get_n_m(obj._nodes[NE], y2_min, y2_max,
					               zoom - 1, y1_count >> 1, test_f,
					               result + num, buffer);
					num += get_n_m(obj._nodes[NW], y2_min, y2_max,
					               zoom - 1, y1_count >> 1, test_f,
					               result + num, buffer);
				}
				if (y2_min < obj._y2_mid_value) {
					num += get_n_m(obj._nodes[SE], y2_min, y2_max,
					               zoom - 1, y1_count >> 1, test_f,
					               result + num, buffer);
					num += get_n_m(obj._nodes[SW], y2_min, y2_max,
					               zoom - 1, y1_count >> 1, test_f,
					               result + num, buffer);
				}
			} else if (obj._datas.size() != 0) {
				/* this is a unsplitted node with data and an array of nxm
				 * entries is needed
				 */
				num = extract(obj, y2_min, y2_max, zoom, y1_count,
				              test_f, result, buffer);
			}
			return num;
		}

		static size_t get_1_m(PVQuadTree const& obj,
		                      const uint64_t y2_min, const uint64_t y2_max,
		                      const uint32_t y1_count,
		                      const test_entry_f &test_f, PVQuadTreeEntry *result,
		                      pv_quadtree_buffer_entry_t *buffer)
		{
			size_t num = 0;
			if (y1_count == 1) {
				/* time to extract
				 */
				result->idx = UINT32_MAX;
				if (get_1_1(obj, y2_min, y2_max, test_f, result)) {
					num = 1;
				}
			} else  if (obj._nodes != 0) {
				if (obj._y2_mid_value < y2_max) {
					num += get_1_m(obj._nodes[NE], y2_min, y2_max,
					               y1_count >> 1, test_f,
					               result + num, buffer);
					num += get_1_m(obj._nodes[NW], y2_min, y2_max,
					               y1_count >> 1, test_f,
					               result + num, buffer);
				}
				if (y2_min < obj._y2_mid_value) {
					num += get_1_m(obj._nodes[SE], y2_min, y2_max,
					               y1_count >> 1, test_f,
					               result + num, buffer);
					num += get_1_m(obj._nodes[SW], y2_min, y2_max,
					               y1_count >> 1, test_f,
					               result + num, buffer);
				}
			} else if (obj._datas.size() != 0) {
				/* this is a unsplitted node with data and an array of 1xm
				 * entries is needed
				 */
				num = extract(obj, y2_min, y2_max, 0, y1_count,
				              test_f, result, buffer);
			}
			return num;
		}

		static size_t get_n_1(PVQuadTree const& obj,
		                      const uint64_t y2_min, const uint64_t y2_max,
		                      const uint32_t zoom,
		                      const test_entry_f &test_f, PVQuadTreeEntry *result,
		                      pv_quadtree_buffer_entry_t *buffer)
		{
			size_t num = 0;
			if (zoom == 0) {
				/* time to extract
				 */
				result->idx = UINT32_MAX;
				if (get_1_1(obj, y2_min, y2_max, test_f, result)) {
					num = 1;
				}
			} else if (obj._nodes != 0) {
				if (obj._y2_mid_value < y2_max) {
					num += get_n_1(obj._nodes[NE], y2_min, y2_max,
					               zoom - 1, test_f,
					               result + num, buffer);
					num += get_n_1(obj._nodes[NW], y2_min, y2_max,
					               zoom - 1, test_f,
					               result + num, buffer);
				}
				if (y2_min < obj._y2_mid_value) {
					num += get_n_1(obj._nodes[SE], y2_min, y2_max,
					               zoom - 1, test_f,
					               result + num, buffer);
					num += get_n_1(obj._nodes[SW], y2_min, y2_max,
					               zoom - 1, test_f,
					               result + num, buffer);
				}
			} else if (obj._datas.size() != 0) {
				/* this is a unsplitted node with data and an array of nx1
				 * entries is needed
				 */
				num = extract(obj, y2_min, y2_max, zoom, 1,
				              test_f, result, buffer);
			}
			return num;
		}

		static bool get_1_1(PVQuadTree const& obj,
		                    const uint64_t y2_min, const uint64_t y2_max,
		                    const test_entry_f &test_f, PVQuadTreeEntry *result)
		{
			bool ret = false;
			if (obj._nodes != 0) {
				/* pick the first relevant element from the children
				 */
				if (obj._y2_mid_value < y2_max) {
					ret |= get_1_1(obj._nodes[NE], y2_min, y2_max, test_f, result);
					ret |= get_1_1(obj._nodes[SE], y2_min, y2_max, test_f, result);
				}
				if (y2_min < obj._y2_mid_value) {
					ret |= get_1_1(obj._nodes[NW], y2_min, y2_max, test_f, result);
					ret |= get_1_1(obj._nodes[SW], y2_min, y2_max, test_f, result);
				}
			} else {
				/* pick the first relevant element from the entry list
				 */
				for (size_t i = 0; i < obj._datas.size(); ++i) {
					const PVQuadTreeEntry &e = obj._datas.at(i);
					if (test_f(e) && (e.idx < result->idx)) {
						*result = e;
						return true;
					}
				}
			}
			return ret;
		}

		static size_t extract(PVQuadTree const& obj,
		                      const uint64_t y2_min, const uint64_t y2_max,
		                      const uint32_t zoom, const uint32_t y1_count,
		                      const test_entry_f &test_f, PVQuadTreeEntry *result,
		                      pv_quadtree_buffer_entry_t *buffer)
		{
			size_t num = 0;
			const uint64_t max_count = 1 << zoom;
			const uint64_t y1_orig = obj._y1_min_value;
			const uint64_t y1_scale = ((obj._y1_mid_value - y1_orig) * 2) / y1_count;
			const uint64_t y2_orig = obj._y2_min_value;
			const uint64_t y2_len = (obj._y2_mid_value - y2_orig) * 2;
			const uint64_t y2_scale = y2_len / max_count;
			const uint64_t ly2_min = (PVCore::clamp(y2_min, y2_orig, y2_orig + y2_len) - y2_orig) / y2_scale;
			const uint64_t ly2_max = (PVCore::clamp(y2_max, y2_orig, y2_orig + y2_len) - y2_orig) / y2_scale;
			const uint64_t clipped_max_count = PVCore::max(1UL, ly2_max - ly2_min);
			const size_t count_aligned = ((clipped_max_count * y1_count) + 31) / 32;
			memset(buffer, 0, count_aligned * sizeof(uint32_t));
			uint32_t remaining = clipped_max_count * y1_count;
			for(size_t i = 0; i < obj._datas.size(); ++i) {
				const PVQuadTreeEntry &e = obj._datas.at(i);
				if (!test_f(e)) {
					continue;
				}
				const uint32_t pos = (((e.y2 - y2_orig) / y2_scale) - ly2_min) + clipped_max_count * ((e.y1 - y1_orig) / y1_scale);
				if (B_IS_SET(buffer[pos >> 5], pos & 31)) {
					continue;
				}
				result[num] = e;
				B_SET(buffer[pos >> 5], pos & 31);
				++num;
				--remaining;
				if (remaining == 0) {
					break;
				}
			}
			return num;
		}
	};

	void get_subtree_from_y1(PVQuadTree& new_tree, uint32_t y1_min, uint32_t y1_max)
	{
		if(_nodes != 0) {
			new_tree._nodes = new PVQuadTree [4];
			for (int i = 0; i < 4; ++i) {
				new_tree._nodes[i].init(_nodes[i]);
			}
			if(_y1_mid_value < y1_max) {
				_nodes[NE].get_subtree_from_y1(new_tree._nodes[NE], y1_min, y1_max);
				_nodes[SE].get_subtree_from_y1(new_tree._nodes[SE], y1_min, y1_max);
			}
			if(y1_min < _y1_mid_value) {
				_nodes[NW].get_subtree_from_y1(new_tree._nodes[NW], y1_min, y1_max);
				_nodes[SW].get_subtree_from_y1(new_tree._nodes[SW], y1_min, y1_max);
			}
		} else {
			new_tree._datas = _datas;
		}
	}

	void get_subtree_from_y2(PVQuadTree& new_tree, uint32_t y2_min, uint32_t y2_max)
	{
		if(_nodes != 0) {
			new_tree._nodes = new PVQuadTree [4];
			for (int i = 0; i < 4; ++i) {
				new_tree._nodes[i].init(_nodes[i]);
			}
			if(_y2_mid_value < y2_max) {
				_nodes[NW].get_subtree_from_y2(new_tree._nodes[NW], y2_min, y2_max);
				_nodes[NE].get_subtree_from_y2(new_tree._nodes[NE], y2_min, y2_max);
			}
			if(y2_min < _y2_mid_value) {
				_nodes[SW].get_subtree_from_y2(new_tree._nodes[SW], y2_min, y2_max);
				_nodes[SE].get_subtree_from_y2(new_tree._nodes[SE], y2_min, y2_max);
			}
		} else {
			new_tree._datas = _datas;
		}
	}

	void get_subtree_from_y1y2(PVQuadTree& new_tree, uint32_t y1_min, uint32_t y1_max, uint32_t y2_min, uint32_t y2_max)
	{
		if(_nodes != 0) {
			new_tree._nodes = new PVQuadTree [4];
			for (int i = 0; i < 4; ++i) {
				new_tree._nodes[i].init(_nodes[i]);
			}
			if(_y1_mid_value < y1_max) {
				if(_y2_mid_value < y2_max) {
					_nodes[NE].get_subtree_from_y1y2(new_tree._nodes[NE], y1_min, y1_max, y2_min, y2_max);
				}
				if(y2_min < _y2_mid_value) {
					_nodes[SE].get_subtree_from_y1y2(new_tree._nodes[SE], y1_min, y1_max, y2_min, y2_max);
				}
			}
			if(y1_min < _y1_mid_value) {
				if(_y2_mid_value < y2_max) {
					_nodes[NW].get_subtree_from_y1y2(new_tree._nodes[NW], y1_min, y1_max, y2_min, y2_max);
				}
				if(y2_min < _y2_mid_value) {
					_nodes[SW].get_subtree_from_y1y2(new_tree._nodes[SW], y1_min, y1_max, y2_min, y2_max);
				}
			}
		} else {
			new_tree._datas = _datas;
		}
	}

	size_t compute_selection_y1(PVQuadTree const& obj, const uint64_t y1_min, const uint64_t y1_max, Picviz::PVSelection &selection) const
	{
		size_t num = 0;
		if (obj._nodes != 0) {
			if (obj._y1_mid_value < y1_max) {
				num += compute_selection_y1(obj._nodes[NE], y1_min, y1_max, selection);
				num += compute_selection_y1(obj._nodes[SE], y1_min, y1_max, selection);
			}
			if (y1_min < obj._y1_mid_value) {
				num += compute_selection_y1(obj._nodes[NW], y1_min, y1_max, selection);
				num += compute_selection_y1(obj._nodes[SW], y1_min, y1_max, selection);
			}
		} else {
			for (size_t i = 0; i < obj._datas.size(); ++i) {
				const PVQuadTreeEntry &e = obj._datas.at(i);
				if ((y1_min <= e.y1) && (e.y1 < y1_max)) {
					selection.set_bit_fast(e.idx);
					++num;
				}
			}
		}
		return num;
	}

	size_t compute_selection_y2(PVQuadTree const& obj, const uint64_t y2_min, const uint64_t y2_max, Picviz::PVSelection &selection) const
	{
		size_t num = 0;
		if (obj._nodes != 0) {
			if (obj._y2_mid_value < y2_max) {
				num += compute_selection_y2(obj._nodes[NE], y2_min, y2_max, selection);
				num += compute_selection_y2(obj._nodes[NW], y2_min, y2_max, selection);
			}
			if (y2_min < obj._y2_mid_value) {
				num += compute_selection_y2(obj._nodes[SE], y2_min, y2_max, selection);
				num += compute_selection_y2(obj._nodes[SW], y2_min, y2_max, selection);
			}
		} else {
			for (size_t i = 0; i < obj._datas.size(); ++i) {
				const PVQuadTreeEntry &e = obj._datas.at(i);
				if ((y2_min <= e.y2) && (e.y2 < y2_max)) {
					selection.set_bit_fast(e.idx);
					++num;
				}
			}
		}
		return num;
	}

private:
	pvquadtree_entries_t  _datas;
	PVQuadTree           *_nodes;

	uint32_t              _y1_min_value;
	uint32_t              _y1_mid_value;
	uint32_t              _y2_min_value;
	uint32_t              _y2_mid_value;

	uint32_t              _max_level;
};

#undef SW
#undef SE
#undef NW
#undef NE

}

#endif // PARALLELVIEW_PVQUADTREE_H
