/**
 * \file PVQuadTree.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PARALLELVIEW_PVQUADTREE_H
#define PARALLELVIEW_PVQUADTREE_H

#include <pvbase/types.h>

#include <pvkernel/core/PVHSVColor.h>

#include <picviz/PVSelection.h>
#include <picviz/PVVector.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCICode.h>

#include <pvkernel/core/PVAllocators.h>

/* TODO: remove all useless code!
 *
 * TODO: make a type for the bitfield visit_{y1,y2}_v2::f::'entries and put it
 *       in ZZT_context and add a PVQuadTree::set_bitfield
 */

// #define QUADTREE_USE_BITFIELD

#define QUADTREE_BF_TEST(BF, I) (BF[(I) >> 5] & ((I) & 31))
#define QUADTREE_BF_SET(BF, I) BF[(I) >> 5] |= (1 << ((I) & 31))

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

// typedef Picviz::PVVector<PVQuadTreeEntry, tbb::scalable_allocator<PVQuadTreeEntry> > pvquadtree_entries_t;
// typedef Picviz::PVVector<PVQuadTreeEntry, 1000, PVCore::PVJEMallocAllocator<PVQuadTreeEntry> > pvquadtree_entries_t;
typedef Picviz::PVVector<PVQuadTreeEntry> pvquadtree_entries_t;

//typedef std::vector<PVParallelView::PVBCICode> pvquadtree_bcicodes_t;

namespace __impl {
	template <typename RESULT>
	struct f_traverse_dim
	{
		typedef size_t(*function_type)(const PVQuadTreeEntry &,
		                               uint32_t y_start,
		                               uint32_t, uint32_t,
		                               const PVCore::PVHSVColor *,
		                               RESULT *);
	};

	template <typename RESULT>
	struct f_traverse_sel
	{
		typedef size_t(*function_type)(const pvquadtree_entries_t &,
		                               const Picviz::PVSelection &,
		                               const PVCore::PVHSVColor *,
		                               RESULT *);
	};

	size_t f_get_first(const PVQuadTreeEntry &e,
	                   uint32_t y_start,
	                   uint32_t shift, uint32_t mask,
	                   const PVCore::PVHSVColor *colors,
	                   PVQuadTreeEntry *entries);

	template <size_t Bbits>
	size_t f_get_first_bci(const PVQuadTreeEntry &e,
	                       uint32_t y_start,
	                       uint32_t shift, uint32_t mask,
	                       const PVCore::PVHSVColor *colors,
	                       PVBCICode<Bbits> *code)
	{
		code->s.idx = e.idx;
		code->s.l = ((e.y1 - y_start) >> shift) & mask;
		code->s.r = ((e.y2 - y_start) >> shift) & mask;
		code->s.color = colors[e.idx].h();
		return 1;
	}

	template <size_t Bbits>
	size_t f_get_first_bci_sel(const pvquadtree_entries_t &entries,
	                           const Picviz::PVSelection &selection,
	                           const PVCore::PVHSVColor *colors,
	                           PVBCICode<Bbits> *code)
	{
		for(unsigned i = 0; i < entries.size(); ++i) {
			const PVQuadTreeEntry &e = entries.at(i);
			if(selection.get_line(e.idx)) {
				code->s.idx = e.idx;
				code->s.l = e.y1 >> (32 - Bbits);
				code->s.r = e.y2 >> (32 - Bbits);
				code->s.color = colors[e.idx].h();
				return 1;
			}
		}
		return 0;
	}

	void f_get_entry_sel(const pvquadtree_entries_t &entries,
	                     const Picviz::PVSelection &selection,
	                     const PVCore::PVHSVColor *colors,
	                     pvquadtree_entries_t &result);

#if 0
	template <size_t Bbits>
	void f_get_bci_sel(const pvquadtree_entries_t &entries,
	                   const Picviz::PVSelection &selection,
	                   const PVCore::PVHSVColor *colors,
	                   pvquadtree_bcicodes_t &result)
	{
		for(unsigned i = 0; i < entries.size(); ++i) {
			const PVQuadTreeEntry &e = entries.at(i);
			if(selection.get_line(e.idx)) {
				PVParallelView::PVBCICode<Bbits> code;
				code.s.idx = e.idx;
				code.s.l = e.y1 >> (32 - Bbits);
				code.s.r = e.y2 >> (32 - Bbits);
				code.s.color = colors[e.idx].h();
				result.push_back(code);
			}
		}
	}
#endif
}

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
			_datas.reserve(_datas.size());
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
	                                PVQuadTreeEntry *result) const
	{
		return visit_y1_v2::f(*this, y1_min, y1_max, zoom,
		                      [&](const PVQuadTreeEntry &e) -> bool
		                      {
			                      return (e.y1 >= y1_min) && (e.y1 < y1_max);
		                      },
		                      result);
	}


	inline size_t get_first_from_y2(uint64_t y2_min, uint64_t y2_max, uint32_t zoom,
	                                PVQuadTreeEntry *result) const
	{
		return visit_y2_v2::f(*this, y2_min, y2_max, zoom,
		                      [&](const PVQuadTreeEntry &e) -> bool
		                      {
			                      return (e.y2 >= y2_min) && (e.y2 < y2_max);
		                      },
		                      result);
	}


	inline size_t get_first_sel_from_y1(uint64_t y1_min, uint64_t y1_max,
	                                    const Picviz::PVSelection &selection,
	                                    uint32_t zoom,
	                                    PVQuadTreeEntry *result) const
	{
		return visit_y1_v2::f(*this, y1_min, y1_max, zoom,
		                      [&](const PVQuadTreeEntry &e) -> bool
		                      {
			                      return (e.y1 >= y1_min) && (e.y1 < y1_max)
				                      && selection.get_line(e.idx);
		                      },
		                      result);
	}


	inline size_t get_first_sel_from_y2(uint64_t y2_min, uint64_t y2_max,
	                                    const Picviz::PVSelection &selection,
	                                    uint32_t zoom,
	                                    PVQuadTreeEntry *result) const
	{
		return visit_y2_v2::f(*this, y2_min, y2_max, zoom,
		                      [&](const PVQuadTreeEntry &e) -> bool
		                      {
			                      return (e.y2 >= y2_min) && (e.y2 < y2_max)
				                      && selection.get_line(e.idx);
		                      },
		                      result);
	}


	inline size_t get_first_bci_from_y1(uint64_t y1_min, uint64_t y1_max, uint32_t zoom, const PVCore::PVHSVColor *colors, PVBCICode<Bbits> *codes) const
	{
		const uint32_t shift = (32 - Bbits) - zoom;
		return visit_y1<PVBCICode<Bbits>, __impl::f_get_first_bci<Bbits>>::f(*this, y1_min, y1_max, zoom, shift, mask_int_ycoord, colors, codes);
	}

	inline size_t get_first_bci_from_y2(uint64_t y2_min, uint64_t y2_max, uint32_t zoom, const PVCore::PVHSVColor *colors, PVBCICode<Bbits> *codes) const
	{
		const uint32_t shift = (32 - Bbits) - zoom;
		return visit_y2<PVBCICode<Bbits>, __impl::f_get_first_bci<Bbits>>::f(*this, y2_min, y2_max, zoom, shift, mask_int_ycoord, colors, codes);
	}

	inline size_t get_first_bci_from_y1y2(uint32_t y1_min, uint32_t y1_max, uint32_t y2_min, uint32_t y2_max, uint32_t zoom, const PVCore::PVHSVColor *colors, PVBCICode<Bbits> *codes) const
	{
		const uint32_t shift = (32 - Bbits) - zoom;
		return visit_y1y2<PVBCICode<Bbits>, __impl::f_get_first_bci<Bbits>>::f(*this, y1_min, y1_max, y2_min, y2_max, zoom, shift, mask_int_ycoord, colors, codes);
	}

	inline size_t get_first_bci_from_selection(const Picviz::PVSelection &selection, const PVCore::PVHSVColor *colors, PVBCICode<Bbits> *codes) const
	{
		return visit_sel<PVBCICode<Bbits>, __impl::f_get_first_bci_sel<Bbits>>::f(*this, selection, colors, codes);
	}

	inline size_t get_first_bci_from_y1_and_selection(uint32_t y1_min, uint32_t y1_max, const Picviz::PVSelection &selection, const PVCore::PVHSVColor *colors, PVBCICode<Bbits> *codes) const
	{
		return visit_y1_sel<PVBCICode<Bbits>, __impl::f_get_first_bci_sel>::f(*this, y1_min, y1_max, selection, colors, codes);
	}

	inline size_t get_first_bci_from_y2_and_selection(uint32_t y2_min, uint32_t y2_max, const Picviz::PVSelection &selection, const PVCore::PVHSVColor *colors, PVBCICode<Bbits> *codes) const
	{
		return visit_y2_sel<PVBCICode<Bbits>, __impl::f_get_first_bci_sel>::f(*this, y2_min, y2_max, selection, colors, codes);
	}

	inline size_t get_first_bci_from_y1y2_and_selection(uint32_t y1_min, uint32_t y1_max, uint32_t y2_min, uint32_t y2_max, const Picviz::PVSelection &selection, const PVCore::PVHSVColor *colors, PVBCICode<Bbits> * codes) const
	{
		return visit_y1y2_sel<PVBCICode<Bbits>, __impl::f_get_first_bci_sel>::f(*this, y1_min, y1_max, y2_min, y2_max, selection, colors, codes);
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
	template <typename RESULT, typename __impl::f_traverse_dim<RESULT>::function_type F>
	struct visit_y1
	{
		static size_t f(PVQuadTree const& obj, uint64_t y1_min, uint64_t y1_max, uint32_t zoom, uint32_t shift, uint32_t mask, const PVCore::PVHSVColor *colors, RESULT *codes)
		{
			if (zoom == 0) {
				if (obj._nodes != 0) {
					// we must get the better first elements from children
					PVQuadTreeEntry e;
					e.idx = UINT_MAX;
					f2(obj, y1_min, y1_max, e);
					if (e.idx != UINT_MAX) {
						// it has been found
						return F(e, y1_min, shift, mask, colors, codes);
					}
				} else {
					// get the first relevant element
					for (size_t i = 0; i < obj._datas.size(); ++i) {
						const PVQuadTreeEntry &e = obj._datas.at(i);
						if ((e.y1 >= y1_min) && (e.y1 < y1_max)) {
							return F(e, y1_min, shift, mask, colors, codes);
						}
					}
				}
				return 0;
			} else {
				size_t num = 0;
				if (obj._nodes != 0) {
					if (obj._y1_mid_value < y1_max) {
						num += f(obj._nodes[NE], y1_min, y1_max, zoom - 1, shift, mask, colors, codes + num);
						num += f(obj._nodes[SE], y1_min, y1_max, zoom - 1, shift, mask, colors, codes + num);
					}
					if (y1_min < obj._y1_mid_value) {
						num += f(obj._nodes[NW], y1_min, y1_max, zoom - 1, shift, mask, colors, codes + num);
						num += f(obj._nodes[SW], y1_min, y1_max, zoom - 1, shift, mask, colors, codes + num);
					}
				} else {
					// we have to extract the 'zoom' first relevant elements from _datas
					// NOTE: the elements should be uniformly distributed, isn't it?
					size_t i = 0, n = 0;
					while ((n < zoom) && (i < obj._datas.size())) {
						const PVQuadTreeEntry &e = obj._datas.at(i);
						if ((e.y1 >= y1_min) && (e.y1 < y1_max)) {
							num += F(e, y1_min, shift, mask, colors, codes + num);
							++n;
						}
						++i;
					}
				}
				return num;
			}
		}
		static void f2(PVQuadTree const& obj, uint64_t y1_min, uint64_t y1_max, PVQuadTreeEntry &result)
		{
			if (obj._nodes != 0) {
				if (obj._y1_mid_value < y1_max) {
					f2(obj._nodes[NE], y1_min, y1_max, result);
					f2(obj._nodes[SE], y1_min, y1_max, result);
				}
				if (y1_min < obj._y1_mid_value) {
					f2(obj._nodes[NW], y1_min, y1_max, result);
					f2(obj._nodes[SW], y1_min, y1_max, result);
				}
			} else {
				for (size_t i = 0; i < obj._datas.size(); ++i) {
					const PVQuadTreeEntry &e = obj._datas.at(i);
					if ((e.y1 >= y1_min) && (e.y1 < y1_max)) {
						if (e.idx <= result.idx) {
							result = e;
						}
					}
				}
			}
		}
	};

	struct visit_y1_v2
	{
		static size_t f(PVQuadTree const& obj,
		                const uint64_t &y1_min, const uint64_t &y1_max, const uint32_t zoom,
		                const test_entry_f &test_f, PVQuadTreeEntry *result)
		{
			if (zoom == 0) {
				if (obj._nodes != 0) {
					// we must get the better first elements from children
					PVQuadTreeEntry e;
					e.idx = UINT_MAX;
					f2(obj, y1_min, y1_max, test_f, e);
					if (e.idx != UINT_MAX) {
						// it has been found
						*result = e;
						return 1;
					}
				} else {
					// get the first relevant element
					for (size_t i = 0; i < obj._datas.size(); ++i) {
						const PVQuadTreeEntry &e = obj._datas.at(i);
						if (test_f(e)) {
							*result = e;
							return 1;
						}
					}
				}
				return 0;
			} else {
				size_t num = 0;
				if (obj._nodes != 0) {
					if (obj._y1_mid_value < y1_max) {
						num += f(obj._nodes[NE], y1_min, y1_max, zoom - 1, test_f, result + num);
						num += f(obj._nodes[SE], y1_min, y1_max, zoom - 1, test_f, result + num);
					}
					if (y1_min < obj._y1_mid_value) {
						num += f(obj._nodes[NW], y1_min, y1_max, zoom - 1, test_f, result + num);
						num += f(obj._nodes[SW], y1_min, y1_max, zoom - 1, test_f, result + num);
					}
				} else {
#ifdef QUADTREE_USE_BITFIELD
					if (obj._datas.size() != 0) {
						int count = 1 << zoom;
						const uint32_t y1_orig = obj._y1_min_value;
						const uint32_t y1_scale = ((obj._y1_mid_value - y1_orig) * 2) / count;

						const int count_aligned = (count + 31) / 32;
						uint32_t entries[count_aligned];
						memset(entries, 0, count_aligned * sizeof(uint32_t));

						for(size_t i = 0; i < obj._datas.size(); ++i) {
							const PVQuadTreeEntry &e = obj._datas.at(i);
							if (!test_f(e)) {
								continue;
							}
							const uint32_t pos = (e.y1 - y1_orig) / y1_scale;
							if (QUADTREE_BF_TEST(entries, pos)) {
								continue;
							}
							result[num] = e;
							QUADTREE_BF_SET(entries, pos);
							++num;
							--count;
							if (count == 0) {
								break;
							}
						}
					}
#else // with PVQuadTreeEntry
					if (obj._datas.size() != 0) {
						int count = 1 << zoom;
						const uint32_t y1_orig = obj._y1_min_value;
						const uint32_t y1_scale = ((obj._y1_mid_value - y1_orig) * 2) / count;

						PVQuadTreeEntry temp_entries[count];
						memset(temp_entries, -1, count * sizeof (PVQuadTreeEntry));

						int remaining = count;
						for(size_t i = 0; i < obj._datas.size(); ++i) {
							const PVQuadTreeEntry &e = obj._datas.at(i);
							if (!test_f(e)) {
								continue;
							}
							const uint32_t pos = (e.y1 - y1_orig) / y1_scale;
							if (temp_entries[pos].idx != UINT_MAX) {
								continue;
							}
							temp_entries[pos] = e;
							--remaining;
							if (remaining == 0) {
								break;
							}
						}

						for(size_t i = 0; i < count; ++i) {
							if (temp_entries[i].idx != UINT_MAX) {
								result[num] = temp_entries[i];
								++num;
							}
						}
					}
#endif
				}
				return num;
			}
		}
		static void f2(PVQuadTree const& obj,
		               const uint64_t &y1_min, const uint64_t &y1_max,
		               const test_entry_f &test_f, PVQuadTreeEntry &result)
		{
			if (obj._nodes != 0) {
				if (obj._y1_mid_value < y1_max) {
					f2(obj._nodes[NE], y1_min, y1_max, test_f, result);
					f2(obj._nodes[SE], y1_min, y1_max, test_f, result);
				}
				if (y1_min < obj._y1_mid_value) {
					f2(obj._nodes[NW], y1_min, y1_max, test_f, result);
					f2(obj._nodes[SW], y1_min, y1_max, test_f, result);
				}
			} else {
				for (size_t i = 0; i < obj._datas.size(); ++i) {
					const PVQuadTreeEntry &e = obj._datas.at(i);
					if (test_f(e) && (e.idx <= result.idx)) {
						result = e;
					}
				}
			}
		}
	};

	template <typename RESULT, typename __impl::f_traverse_dim<RESULT>::function_type F>
	struct visit_y2
	{
		static size_t f(PVQuadTree const& obj, uint64_t y2_min, uint64_t y2_max, uint32_t zoom, uint32_t shift, uint32_t mask, const PVCore::PVHSVColor *colors, RESULT *codes)
		{
			if (zoom == 0) {
				if (obj._nodes != 0) {
					// we must get the better first elements from children
					PVQuadTreeEntry e;
					e.idx = UINT_MAX;
					f2(obj, y2_min, y2_max, e);
					if (e.idx != UINT_MAX) {
						// it has been found
						return F(e, y2_min, shift, mask, colors, codes);
					}
				} else {
					// get the first relevant element
					for (size_t i = 0; i < obj._datas.size(); ++i) {
						const PVQuadTreeEntry &e = obj._datas.at(i);
						if ((e.y2 >= y2_min) && (e.y2 < y2_max)) {
							return F(e, y2_min, shift, mask, colors, codes);
						}
					}
				}
				return 0;
			} else {
				size_t num = 0;
				if (obj._nodes != 0) {
					if (obj._y2_mid_value < y2_max) {
						num += f(obj._nodes[NE], y2_min, y2_max, zoom - 1, shift, mask, colors, codes + num);
						num += f(obj._nodes[NW], y2_min, y2_max, zoom - 1, shift, mask, colors, codes + num);
					}
					if (y2_min < obj._y2_mid_value) {
						num += f(obj._nodes[SE], y2_min, y2_max, zoom - 1, shift, mask, colors, codes + num);
						num += f(obj._nodes[SW], y2_min, y2_max, zoom - 1, shift, mask, colors, codes + num);
					}
				} else {
					// we have to extract the 'zoom' first relevant elements from _datas
					// NOTE: the elements should be uniformly distributed, isn't it?
					size_t i = 0, n = 0;
					while ((n < zoom) && (i < obj._datas.size())) {
						const PVQuadTreeEntry &e = obj._datas.at(i);
						if ((e.y2 >= y2_min) && (e.y2 < y2_max)) {
							num += F(e, y2_min, shift, mask, colors, codes + num);
							++n;
						}
						++i;
					}
				}
				return num;
			}
		}
		static void f2(PVQuadTree const& obj, uint64_t y2_min, uint64_t y2_max, PVQuadTreeEntry &result)
		{
			if (obj._nodes != 0) {
				if (obj._y2_mid_value < y2_max) {
					f2(obj._nodes[NE], y2_min, y2_max, result);
					f2(obj._nodes[NW], y2_min, y2_max, result);
				}
				if (y2_min < obj._y2_mid_value) {
					f2(obj._nodes[SE], y2_min, y2_max, result);
					f2(obj._nodes[SW], y2_min, y2_max, result);
				}
			} else {
				for (size_t i = 0; i < obj._datas.size(); ++i) {
					const PVQuadTreeEntry &e = obj._datas.at(i);
					if ((e.y2 >= y2_min) && (e.y2 < y2_max)) {
						if (e.idx <= result.idx) {
							result = e;
						}
					}
				}
			}
		}
	};

	struct visit_y2_v2
	{
		static size_t f(PVQuadTree const& obj,
		                const uint64_t &y2_min, const uint64_t &y2_max, const uint32_t zoom,
		                const test_entry_f &test_f, PVQuadTreeEntry *result)
		{
			if (zoom == 0) {
				if (obj._nodes != 0) {
					// we must get the better first elements from children
					PVQuadTreeEntry e;
					e.idx = UINT_MAX;
					f2(obj, y2_min, y2_max, test_f, e);
					if (e.idx != UINT_MAX) {
						// it has been found
						*result = e;
						return 1;
					}
				} else {
					// get the first relevant element
					for (size_t i = 0; i < obj._datas.size(); ++i) {
						const PVQuadTreeEntry &e = obj._datas.at(i);
						if (test_f(e)) {
							*result = e;
							return 1;
						}
					}
				}
				return 0;
			} else {
				size_t num = 0;
				if (obj._nodes != 0) {
					if (obj._y2_mid_value < y2_max) {
						num += f(obj._nodes[NE], y2_min, y2_max, zoom - 1, test_f, result + num);
						num += f(obj._nodes[NW], y2_min, y2_max, zoom - 1, test_f, result + num);
					}
					if (y2_min < obj._y2_mid_value) {
						num += f(obj._nodes[SE], y2_min, y2_max, zoom - 1, test_f, result + num);
						num += f(obj._nodes[SW], y2_min, y2_max, zoom - 1, test_f, result + num);
					}
				} else {
#ifdef QUADTREE_USE_BITFIELD
					if (obj._datas.size() != 0) {
						int count = 1 << zoom;
						const uint32_t y2_orig = obj._y2_min_value;
						const uint32_t y2_scale = ((obj._y2_mid_value - y2_orig) * 2) / count;

						const int count_aligned = (count + 31) / 32;
						uint32_t entries[count_aligned];
						memset(entries, 0, count_aligned * sizeof(uint32_t));

						for(size_t i = 0; i < obj._datas.size(); ++i) {
							const PVQuadTreeEntry &e = obj._datas.at(i);
							if (!test_f(e)) {
								continue;
							}
							const uint32_t pos = (e.y2 - y2_orig) / y2_scale;
							if (QUADTREE_BF_TEST(entries, pos)) {
								continue;
							}
							result[num] = e;
							QUADTREE_BF_SET(entries, pos);
							++num;
							--count;
							if (count == 0) {
								break;
							}
						}
					}
#else // with PVQuadTreeEntry
					if (obj._datas.size() != 0) {
						int count = 1 << zoom;
						const uint32_t y2_orig = obj._y2_min_value;
						const uint32_t y2_scale = ((obj._y2_mid_value - y2_orig) * 2) / count;

						PVQuadTreeEntry temp_entries[count];
						memset(temp_entries, -1, count * sizeof (PVQuadTreeEntry));

						int remaining = count;
						for(size_t i = 0; i < obj._datas.size(); ++i) {
							const PVQuadTreeEntry &e = obj._datas.at(i);
							if (!test_f(e)) {
								continue;
							}
							const uint32_t pos = (e.y2 - y2_orig) / y2_scale;
							if (temp_entries[pos].idx != UINT_MAX) {
								continue;
							}
							temp_entries[pos] = e;
							--remaining;
							if (remaining == 0) {
								break;
							}
						}

						for(size_t i = 0; i < count; ++i) {
							if (temp_entries[i].idx != UINT_MAX) {
								result[num] = temp_entries[i];
								++num;
							}
						}
					}
#endif
				}
				return num;
			}
		}
		static void f2(PVQuadTree const& obj,
		               const uint64_t &y2_min, const uint64_t &y2_max,
		               const test_entry_f &test_f, PVQuadTreeEntry &result)
		{
			if (obj._nodes != 0) {
				if (obj._y2_mid_value < y2_max) {
					f2(obj._nodes[NE], y2_min, y2_max, test_f, result);
					f2(obj._nodes[NW], y2_min, y2_max, test_f, result);
				}
				if (y2_min < obj._y2_mid_value) {
					f2(obj._nodes[SE], y2_min, y2_max, test_f, result);
					f2(obj._nodes[SW], y2_min, y2_max, test_f, result);
				}
			} else {
				for (size_t i = 0; i < obj._datas.size(); ++i) {
					const PVQuadTreeEntry &e = obj._datas.at(i);
					if (test_f(e) && (e.idx <= result.idx)) {
						result = e;
					}
				}
			}
		}
	};

	template <typename RESULT, typename __impl::f_traverse_dim<RESULT>::function_type F>
	struct visit_y1y2
	{
		static size_t f(PVQuadTree const& obj, uint32_t y1_min, uint32_t y1_max, uint32_t y2_min, uint32_t y2_max, uint32_t zoom, uint32_t shift, uint32_t mask, const PVCore::PVHSVColor *colors, RESULT *codes)
		{
			if (zoom == 0) {
				if (obj._nodes != 0) {
					// we must get the better first elements from children
					PVQuadTreeEntry e;
					e.idx = UINT_MAX;
					f2(obj, y1_min, y1_max, y2_min, y2_max, e);
					if (e.idx != UINT_MAX) {
						// it has been found
						return F(e, shift, mask, colors, codes);
					}
				} else {
					// the first element has been found
					if (obj._datas.size() != 0) {
						return F(obj._datas.at(0), shift, mask, colors, codes);
					}
				}
				return 0;
			} else {
				size_t num = 0;
				if (obj._nodes != 0) {
					if(obj._y1_mid_value < y1_max) {
						if(obj._y2_mid_value < y2_max) {
							num += f(obj._nodes[NE], y1_min, y1_max, y2_min, y2_max, zoom - 1, shift, mask, colors, codes + num);
						}
						if(y2_min < obj._y2_mid_value) {
							num += f(obj._nodes[SE], y1_min, y1_max, y2_min, y2_max, zoom - 1, shift, mask, colors, codes + num);
						}
					}
					if(y1_min < obj._y1_mid_value) {
						if(obj._y2_mid_value < y2_max) {
							num += f(obj._nodes[NW], y1_min, y1_max, y2_min, y2_max, zoom - 1, shift, mask, colors, codes + num);
						}
						if(y2_min < obj._y2_mid_value) {
							num += f(obj._nodes[SW], y1_min, y1_max, y2_min, y2_max, zoom - 1, shift, mask, colors, codes + num);
						}
					}
				} else {
					// we have to extract the 'zoom' first elements from _datas
					for (unsigned i = 0; i < std::min(zoom, obj._datas.size()); ++i) {
						num += F(obj._datas.at(i), shift, mask, colors, codes + num);
					}
				}
				return num;
			}
		}

		static void f2(PVQuadTree const& obj, uint32_t y1_min, uint32_t y1_max, uint32_t y2_min, uint32_t y2_max, PVQuadTreeEntry &result)
		{
			if (obj._nodes != 0) {
				if(obj._y1_mid_value < y1_max) {
					if(obj._y2_mid_value < y2_max) {
						f2(obj._nodes[NE], y1_min, y1_max, y2_min, y2_max, result);
					}
					if(y2_min < obj._y2_mid_value) {
						f2(obj._nodes[SE], y1_min, y1_max, y2_min, y2_max, result);
					}
				}
				if(y1_min < obj._y1_mid_value) {
					if(obj._y2_mid_value < y2_max) {
						f2(obj._nodes[NW], y1_min, y1_max, y2_min, y2_max, result);
					}
					if(y2_min < obj._y2_mid_value) {
						f2(obj._nodes[SW], y1_min, y1_max, y2_min, y2_max, result);
					}
				}
			} else {
				if (obj._datas.size() != 0) {
					const PVQuadTreeEntry &e = obj._datas.at(0);
					if (e.idx < result.idx) {
						result = e;
					}
				}
			}
		}
	};

	template <typename RESULT, typename __impl::f_traverse_sel<RESULT>::function_type F>
	struct visit_sel
	{
		static size_t f(PVQuadTree const& obj, const Picviz::PVSelection &selection, const PVCore::PVHSVColor *colors, RESULT *codes)
		{
			if(obj._nodes != 0) {
				size_t num;
				num = f(obj._nodes[NE], selection, colors, codes);
				num += f(obj._nodes[SE], selection, colors, codes + num);
				num += f(obj._nodes[NW], selection, colors, codes + num);
				num += f(obj._nodes[SW], selection, colors, codes + num);
				return num;
			} else {
				return F(obj._datas, selection, colors, codes);
			}
		}
	};

	template <typename RESULT, typename __impl::f_traverse_sel<RESULT>::function_type F>
	struct visit_y1_sel
	{
		static size_t f(PVQuadTree const& obj, uint32_t y1_min, uint32_t y1_max, const Picviz::PVSelection &selection, const PVCore::PVHSVColor *colors, RESULT *codes)
		{
			if (obj._nodes != 0) {
				size_t num = 0;
				if (obj._y1_mid_value < y1_max) {
					num += f(obj._nodes[NE], y1_min, y1_max, selection, colors, codes + num);
					num += f(obj._nodes[SE], y1_min, y1_max, selection, colors, codes + num);
				}
				if (y1_min < obj._y1_mid_value) {
					num += f(obj._nodes[NW], y1_min, y1_max, selection, colors, codes + num);
					num += f(obj._nodes[SW], y1_min, y1_max, selection, colors, codes + num);
				}
				return num;
			} else {
				return F(obj._datas, selection, colors, codes);
			}
		}
	};

	template <typename RESULT, typename __impl::f_traverse_sel<RESULT>::function_type F>
	struct visit_y2_sel
	{
		static size_t f(PVQuadTree const& obj, uint32_t y2_min, uint32_t y2_max, const Picviz::PVSelection &selection, const PVCore::PVHSVColor *colors, RESULT *codes)
		{
			if (obj._nodes != 0) {
				size_t num = 0;
				if (obj._y2_mid_value < y2_max) {
					num += f(obj._nodes[NE], y2_min, y2_max, selection, colors, codes + num);
					num += f(obj._nodes[NW], y2_min, y2_max, selection, colors, codes + num);
				}
				if (y2_min < obj._y2_mid_value) {
					num += f(obj._nodes[SE], y2_min, y2_max, selection, colors, codes + num);
					num += f(obj._nodes[SW], y2_min, y2_max, selection, colors, codes + num);
				}
				return num;
			} else {
				return F(obj._datas, selection, colors, codes);
			}
		}
	};

	template <typename RESULT, typename __impl::f_traverse_sel<RESULT>::function_type F>
	struct visit_y1y2_sel
	{
		static size_t f(PVQuadTree const& obj, uint32_t y1_min, uint32_t y1_max, uint32_t y2_min, uint32_t y2_max, const Picviz::PVSelection &selection, const PVCore::PVHSVColor *colors, RESULT *codes)
		{
			if (obj._nodes != 0) {
				size_t num = 0;
				if(obj._y1_mid_value < y1_max) {
					if(obj._y2_mid_value < y2_max) {
						num += f(obj._nodes[NE], y1_min, y1_max, y2_min, y2_max, selection, colors, codes + num);
					}
					if(y2_min < obj._y2_mid_value) {
						num += f(obj._nodes[SE], y1_min, y1_max, y2_min, y2_max, selection, colors, codes + num);
					}
				}
				if(y1_min < obj._y1_mid_value) {
					if(obj._y2_mid_value < y2_max) {
						num += f(obj._nodes[NW], y1_min, y1_max, y2_min, y2_max, selection, colors, codes + num);
					}
					if(y2_min < obj._y2_mid_value) {
						num += f(obj._nodes[SW], y1_min, y1_max, y2_min, y2_max, selection, colors, codes + num);
					}
				}
				return num;
			} else {
				return F(obj._datas, selection, colors, codes);
			}
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

	void get_subtree_from_selection(PVQuadTree& new_tree, const Picviz::PVSelection &selection)
	{
		if(_nodes != 0) {
			new_tree._nodes = new PVQuadTree [4];
			for (int i = 0; i < 4; ++i) {
				new_tree._nodes[i].init(_nodes[i]);
				_nodes[i].get_subtree_from_selection(new_tree._nodes[i], selection);
			}
		} else {
			__impl::f_get_entry_sel(_datas, selection, 0, new_tree._datas);
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
				if ((y1_min <= e.y1) && (e.y1 <= y1_max)) {
					selection.set_bit_fast(e.idx);
					++num;
				}
			}
		}
		return num;
	}

	size_t compute_selection_y2(PVQuadTree const& obj, const int64_t y2_min, const int64_t y2_max, Picviz::PVSelection &selection) const
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
				if ((y2_min <= e.y2) && (e.y2 <= y2_max)) {
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
