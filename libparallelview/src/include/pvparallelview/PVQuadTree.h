#ifndef PARALLELVIEW_PVQUADTREE_H
#define PARALLELVIEW_PVQUADTREE_H

#include <pvbase/types.h>

#include <picviz/PVSelection.h>
#include <picviz/PVVector.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVHSVColor.h>

#include <pvkernel/core/PVAllocators.h>

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

typedef std::vector<PVParallelView::PVBCICode> pvquadtree_bcicodes_t;

namespace __impl {
	template <typename RESULT>
	struct f_traverse_dim
	{
		typedef size_t(*function_type)(const PVQuadTreeEntry &,
		                               uint32_t, uint32_t,
		                               const PVHSVColor *,
		                               RESULT *);
	};

	template <typename RESULT>
	struct f_traverse_sel
	{
		typedef size_t(*function_type)(const pvquadtree_entries_t &,
		                               const Picviz::PVSelection &,
		                               const PVHSVColor *,
		                               RESULT *);
	};

	size_t f_get_first_bci(const PVQuadTreeEntry &e,
	                       uint32_t shift, uint32_t mask,
	                       const PVHSVColor *colors,
	                       PVBCICode *code);

	size_t f_get_first_bci_sel(const pvquadtree_entries_t &entries,
	                           const Picviz::PVSelection &selection,
	                           const PVHSVColor *colors,
	                           PVBCICode *code);

	void f_get_entry_sel(const pvquadtree_entries_t &entries,
	                     const Picviz::PVSelection &selection,
	                     const PVHSVColor *colors,
	                     pvquadtree_entries_t &result);

	void f_get_bci_sel(const pvquadtree_entries_t &entries,
	                   const Picviz::PVSelection &selection,
	                   const PVHSVColor *colors,
	                   pvquadtree_bcicodes_t &result);
}

template<int MAX_ELEMENTS_PER_NODE = 10000, int REALLOC_ELEMENT_COUNT = 1000, int PREALLOC_ELEMENT_COUNT = 0>
class PVQuadTree
{
public:
	PVQuadTree(uint32_t y1_min_value, uint32_t y1_max_value, uint32_t y2_min_value, uint32_t y2_max_value, int max_level)
	{
		uint32_t y1_mid = (y1_max_value - y1_min_value) >> 1;
		uint32_t y2_mid = (y2_max_value - y2_min_value) >> 1;

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
		PVQuadTree *qt = this;
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

	void compact()
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

	inline size_t get_first_bci_from_y1(uint32_t y1_min, uint32_t y1_max, uint32_t zoom, const PVHSVColor *colors, PVBCICode *codes) const
	{
		uint32_t shift = (32 - NBITS_INDEX) - zoom;
		return visit_y1<PVBCICode, __impl::f_get_first_bci>::f(*this, y1_min, y1_max, zoom, shift, MASK_INT_YCOORD, colors, codes);
	}

	inline size_t get_first_bci_from_y2(uint32_t y2_min, uint32_t y2_max, uint32_t zoom, const PVHSVColor *colors, PVBCICode *codes) const
	{
		uint32_t shift = (32 - NBITS_INDEX) - zoom;
		return visit_y2<PVBCICode, __impl::f_get_first_bci>::f(*this, y2_min, y2_max, zoom, shift, MASK_INT_YCOORD, colors, codes);
	}

	inline size_t get_first_bci_from_y1y2(uint32_t y1_min, uint32_t y1_max, uint32_t y2_min, uint32_t y2_max, uint32_t zoom, const PVHSVColor *colors, PVBCICode *codes) const
	{
		uint32_t shift = (32 - NBITS_INDEX) - zoom;
		return visit_y1y2<PVBCICode, __impl::f_get_first_bci>::f(*this, y1_min, y1_max, y2_min, y2_max, zoom, shift, MASK_INT_YCOORD, colors, codes);
	}

	inline size_t get_first_bci_from_selection(const Picviz::PVSelection &selection, const PVHSVColor *colors, PVBCICode *codes) const
	{
		return visit_sel<PVBCICode, __impl::f_get_first_bci_sel>::f(*this, selection, colors, codes);
	}

	inline size_t get_first_bci_from_y1_and_selection(uint32_t y1_min, uint32_t y1_max, const Picviz::PVSelection &selection, const PVHSVColor *colors, PVBCICode *codes) const
	{
		return visit_y1_sel<PVBCICode, __impl::f_get_first_bci_sel>::f(*this, y1_min, y1_max, selection, colors, codes);
	}

	inline size_t get_first_bci_from_y2_and_selection(uint32_t y2_min, uint32_t y2_max, const Picviz::PVSelection &selection, const PVHSVColor *colors, PVBCICode *codes) const
	{
		return visit_y2_sel<PVBCICode, __impl::f_get_first_bci_sel>::f(*this, y2_min, y2_max, selection, colors, codes);
	}

	inline size_t get_first_bci_from_y1y2_and_selection(uint32_t y1_min, uint32_t y1_max, uint32_t y2_min, uint32_t y2_max, const Picviz::PVSelection &selection, const PVHSVColor *colors, PVBCICode * codes) const
	{
		return visit_y1y2_sel<PVBCICode, __impl::f_get_first_bci_sel>::f(*this, y1_min, y1_max, y2_min, y2_max, selection, colors, codes);
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
		static size_t f(PVQuadTree const& obj, uint32_t y1_min, uint32_t y1_max, uint32_t zoom, uint32_t shift, uint32_t mask, const PVHSVColor *colors, RESULT *codes)
		{
			if (zoom == 0) {
				if (obj._nodes != 0) {
					// we must get the better first elements from children
					PVQuadTreeEntry e;
					e.idx = UINT_MAX;
					f2(obj, y1_min, y1_max, e);
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
					if (obj._y1_mid_value < y1_max) {
						num += f(obj._nodes[NE], y1_min, y1_max, zoom - 1, shift, mask, colors, codes + num);
						num += f(obj._nodes[SE], y1_min, y1_max, zoom - 1, shift, mask, colors, codes + num);
					}
					if (y1_min < obj._y1_mid_value) {
						num += f(obj._nodes[NW], y1_min, y1_max, zoom - 1, shift, mask, colors, codes + num);
						num += f(obj._nodes[SW], y1_min, y1_max, zoom - 1, shift, mask, colors, codes + num);
					}
				} else {
					// we have to extract the 'zoom' first elements from _datas
					uint32_t count = std::min(zoom, obj._datas.size());
					for (unsigned i = 0; i < count; ++i) {
						const PVQuadTreeEntry &e = obj._datas.at(i);
						num += F(e, shift, mask, colors, codes + num);
					}
				}
				return num;
			}
		}
		static void f2(PVQuadTree const& obj, uint32_t y1_min, uint32_t y1_max, PVQuadTreeEntry &result)
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
				if (obj._datas.size() != 0) {
					const PVQuadTreeEntry &e = obj._datas.at(0);
					if (e.idx <= result.idx) {
						result = e;
					}
				}
			}
		}
	};

	template <typename RESULT, typename __impl::f_traverse_dim<RESULT>::function_type F>
	struct visit_y2
	{
		static size_t f(PVQuadTree const& obj, uint32_t y2_min, uint32_t y2_max, uint32_t zoom, uint32_t shift, uint32_t mask, const PVHSVColor *colors, RESULT *codes)
		{
			if (zoom == 0) {
				if (obj._nodes != 0) {
					// we must get the better first elements from children
					PVQuadTreeEntry e;
					e.idx = UINT_MAX;
					f2(obj, y2_min, y2_max, e);
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
					if (obj._y2_mid_value < y2_max) {
						num += f(obj._nodes[NE], y2_min, y2_max, zoom - 1, shift, mask, colors, codes + num);
						num += f(obj._nodes[NW], y2_min, y2_max, zoom - 1, shift, mask, colors, codes + num);
					}
					if (y2_min < obj._y2_mid_value) {
						num += f(obj._nodes[SE], y2_min, y2_max, zoom - 1, shift, mask, colors, codes + num);
						num += f(obj._nodes[SW], y2_min, y2_max, zoom - 1, shift, mask, colors, codes + num);
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

		static void f2(PVQuadTree const& obj, uint32_t y2_min, uint32_t y2_max, PVQuadTreeEntry &result)
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
				if (obj._datas.size() != 0) {
					const PVQuadTreeEntry &e = obj._datas.at(0);
					if (e.idx <= result.idx) {
						result = e;
					}
				}
			}
		}
	};

	template <typename RESULT, typename __impl::f_traverse_dim<RESULT>::function_type F>
	struct visit_y1y2
	{
		static size_t f(PVQuadTree const& obj, uint32_t y1_min, uint32_t y1_max, uint32_t y2_min, uint32_t y2_max, uint32_t zoom, uint32_t shift, uint32_t mask, const PVHSVColor *colors, RESULT *codes)
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
		static size_t f(PVQuadTree const& obj, const Picviz::PVSelection &selection, const PVHSVColor *colors, RESULT *codes)
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
		static size_t f(PVQuadTree const& obj, uint32_t y1_min, uint32_t y1_max, const Picviz::PVSelection &selection, const PVHSVColor *colors, RESULT *codes)
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
		static size_t f(PVQuadTree const& obj, uint32_t y2_min, uint32_t y2_max, const Picviz::PVSelection &selection, const PVHSVColor *colors, RESULT *codes)
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
		static size_t f(PVQuadTree const& obj, uint32_t y1_min, uint32_t y1_max, uint32_t y2_min, uint32_t y2_max, const Picviz::PVSelection &selection, const PVHSVColor *colors, RESULT *codes)
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
