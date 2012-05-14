#ifndef PARALLELVIEW_PVQUADTREE_H
#define PARALLELVIEW_PVQUADTREE_H

#include <pvbase/types.h>

#include <picviz/PVSelection.h>
#include <picviz/PVVector.h>

#include <pvparallelview/PVBCICode.h>

namespace PVParallelView {

#define PVQUADTREE_MAX_NODE_ELEMENT_COUNT 10000

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
};
#pragma pack(pop)

typedef Picviz::PVVector<PVQuadTreeEntry> pvquadtree_entries_t;
typedef std::vector<PVParallelView::PVBCICode> pvquadtree_bcicodes_t;

namespace __impl {
	template <typename RESULT>
	struct f_traverse_dim
	{
		typedef void(*function_type)(const pvquadtree_entries_t &,
		                             RESULT&);
	};

	template <typename RESULT>
	struct f_traverse_sel
	{
		typedef void(*function_type)(const pvquadtree_entries_t &,
		                             const Picviz::PVSelection &,
		                             RESULT&);
	};

	void f_get_first_entry(const pvquadtree_entries_t &entries,
	                       pvquadtree_entries_t &result)
	{
		if(entries.size() != 0) {
			result.push_back(entries.at(0));
		}
	}

	void f_get_first_bci(const pvquadtree_entries_t &entries,
	                     pvquadtree_bcicodes_t &result)
	{
		if(entries.size() != 0) {
			const PVQuadTreeEntry &e = entries.at(0);
			PVParallelView::PVBCICode code;
			code.s.idx = e.idx;
			code.s.l = (e.y1 >> 22) && 0x3FF;
			code.s.r = (e.y2 >> 22) && 0x3FF;
			code.s.color = random() & 255;
			result.push_back(code);
		}
	}

	void f_get_first_entry_sel(const pvquadtree_entries_t &entries,
	                           const Picviz::PVSelection &selection,
	                           pvquadtree_entries_t &result)
	{
		for(unsigned i = 0; i < entries.size(); ++i) {
			const PVQuadTreeEntry &e = entries.at(i);
			if(selection.get_line(e.idx)) {
				result.push_back(e);
				break;
			}
		}
	}

	void f_get_first_bci_sel(const pvquadtree_entries_t &entries,
	                         const Picviz::PVSelection &selection,
	                         pvquadtree_bcicodes_t &result)
	{
		for(unsigned i = 0; i < entries.size(); ++i) {
			const PVQuadTreeEntry &e = entries.at(i);
			if(selection.get_line(e.idx)) {
				PVParallelView::PVBCICode code;
				code.s.idx = e.idx;
				code.s.l = e.y1 >> 22;
				code.s.r = e.y2 >> 22;
				code.s.color = random() & 255;
				result.push_back(code);
				break;
			}
		}
	}

	void f_get_entry_sel(const pvquadtree_entries_t &entries,
	                     const Picviz::PVSelection &selection,
	                     pvquadtree_entries_t &result)
	{
		for(unsigned i = 0; i < entries.size(); ++i) {
			const PVQuadTreeEntry &e = entries.at(i);
			if(selection.get_line(e.idx)) {
				result.push_back(e);
			}
		}
	}

	void f_get_bci_sel(const pvquadtree_entries_t &entries,
	                   const Picviz::PVSelection &selection,
	                   pvquadtree_bcicodes_t &result)
	{
		for(unsigned i = 0; i < entries.size(); ++i) {
			const PVQuadTreeEntry &e = entries.at(i);
			if(selection.get_line(e.idx)) {
				PVParallelView::PVBCICode code;
				code.s.idx = e.idx;
				code.s.l = e.y1 >> 22;
				code.s.r = e.y2 >> 22;
				code.s.color = random() & 255;
				result.push_back(code);
			}
		}
	}
}

class PVQuadTree
{
public:
	PVQuadTree(uint32_t y1_min_value, uint32_t y1_max_value, uint32_t y2_min_value, uint32_t y2_max_value, int max_level)
	{
		init(y1_min_value, y1_max_value, y2_min_value, y2_max_value, max_level);
	}

	~PVQuadTree() {
		if (_nodes == 0) {
			_datas.clear();
		} else {
			delete [] _nodes;
		}
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
		if ((qt->_datas.size() >= PVQUADTREE_MAX_NODE_ELEMENT_COUNT) && qt->_max_level) {
			qt->create_next_level();
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

	inline void get_first_entry_from_y1(uint32_t y1_min, uint32_t y1_max, pvquadtree_entries_t &result) const
	{
		visit_y1<pvquadtree_entries_t, __impl::f_get_first_entry>::f(*this, y1_min, y1_max, result);
	}

	inline void get_first_entry_from_y2(uint32_t y2_min, uint32_t y2_max, pvquadtree_entries_t &result) const
	{
		visit_y2<pvquadtree_entries_t, __impl::f_get_first_entry>::f(*this, y2_min, y2_max, result);
	}

	inline void get_first_entry_from_y1y2(uint32_t y1_min, uint32_t y1_max, uint32_t y2_min, uint32_t y2_max, pvquadtree_entries_t &result) const
	{
		visit_y1y2<pvquadtree_entries_t, __impl::f_get_first_entry>::f(*this, y1_min, y1_max, y2_min, y2_max, result);
	}

	void get_first_entry_from_selection(const Picviz::PVSelection &selection, pvquadtree_entries_t &result) const
	{
		visit_sel<pvquadtree_entries_t, __impl::f_get_first_entry_sel>::f(*this, selection, result);
	}

	inline void get_first_from_y1_and_selection(uint32_t y1_min, uint32_t y1_max, const Picviz::PVSelection &selection, pvquadtree_entries_t &result) const
	{
		visit_y1_sel<pvquadtree_entries_t, __impl::f_get_first_entry_sel>::f(*this, y1_min, y1_max, selection, result);
	}

	inline void get_first_from_y2_and_selection(uint32_t y2_min, uint32_t y2_max, const Picviz::PVSelection &selection, pvquadtree_entries_t &result) const
	{
		visit_y2_sel<pvquadtree_entries_t, __impl::f_get_first_entry_sel>::f(*this, y2_min, y2_max, selection, result);
	}

	inline void get_first_from_y1y2_and_selection(uint32_t y1_min, uint32_t y1_max, uint32_t y2_min, uint32_t y2_max, const Picviz::PVSelection &selection, pvquadtree_entries_t &result) const
	{
		visit_y1y2_sel<pvquadtree_entries_t, __impl::f_get_first_entry_sel>::f(*this, y1_min, y1_max, y2_min, y2_max, selection, result);
	}

	inline void get_first_bci_from_y1(uint32_t y1_min, uint32_t y1_max, pvquadtree_bcicodes_t &result) const
	{
		visit_y1<pvquadtree_bcicodes_t, __impl::f_get_first_bci>::f(*this, y1_min, y1_max, result);
	}

	inline void get_first_bci_from_y2(uint32_t y2_min, uint32_t y2_max, pvquadtree_bcicodes_t &result) const
	{
		visit_y2<pvquadtree_bcicodes_t, __impl::f_get_first_bci>::f(*this, y2_min, y2_max, result);
	}

	inline void get_first_bci_from_y1y2(uint32_t y1_min, uint32_t y1_max, uint32_t y2_min, uint32_t y2_max, pvquadtree_bcicodes_t &result) const
	{
		visit_y1y2<pvquadtree_bcicodes_t, __impl::f_get_first_bci>::f(*this, y1_min, y1_max, y2_min, y2_max, result);
	}

	void get_first_bci_from_selection(const Picviz::PVSelection &selection, pvquadtree_bcicodes_t &result) const
	{
		visit_sel<pvquadtree_bcicodes_t, __impl::f_get_first_bci_sel>::f(*this, selection, result);
	}

	inline void get_first_bci_from_y1_and_selection(uint32_t y1_min, uint32_t y1_max, const Picviz::PVSelection &selection, pvquadtree_bcicodes_t &result) const
	{
		visit_y1_sel<pvquadtree_bcicodes_t, __impl::f_get_first_bci_sel>::f(*this, y1_min, y1_max, selection, result);
	}

	inline void get_first_bci_from_y2_and_selection(uint32_t y2_min, uint32_t y2_max, const Picviz::PVSelection &selection, pvquadtree_bcicodes_t &result) const
	{
		visit_y2_sel<pvquadtree_bcicodes_t, __impl::f_get_first_bci_sel>::f(*this, y2_min, y2_max, selection, result);
	}

	inline void get_first_bci_from_y1y2_and_selection(uint32_t y1_min, uint32_t y1_max, uint32_t y2_min, uint32_t y2_max, const Picviz::PVSelection &selection, pvquadtree_bcicodes_t &result) const
	{
		visit_y1y2_sel<pvquadtree_bcicodes_t, __impl::f_get_first_bci_sel>::f(*this, y1_min, y1_max, y2_min, y2_max, selection, result);
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
	// CTOR to use with call to init()
	PVQuadTree()
	{
	}

	void init(const PVQuadTree &qt)
	{
		_y1_min_value = qt._y1_min_value;
		_y1_max_value = qt._y1_max_value;
		_y2_min_value = qt._y2_min_value;
		_y2_max_value = qt._y2_max_value;
		_y1_mid_value = qt._y1_mid_value;
		_y2_mid_value = qt._y2_mid_value;
		_max_level = qt._max_level;
		_nodes = 0;
	}

	void init(uint32_t y1_min_value, uint32_t y1_max_value, uint32_t y2_min_value, uint32_t y2_max_value, int max_level)
	{
		_y1_min_value = y1_min_value;
		_y1_max_value = y1_max_value;
		_y2_min_value = y2_min_value;
		_y2_max_value = y2_max_value;
		_max_level = max_level;
		_y1_mid_value = (_y1_min_value + _y1_max_value) / 2;
		_y2_mid_value = (_y2_min_value + _y2_max_value) / 2;
		// + 1 to avoid reallocating before a split occurs
		_datas.reserve(PVQUADTREE_MAX_NODE_ELEMENT_COUNT + 1);
		_nodes = 0;
	}

	inline int compute_index(const PVQuadTreeEntry &e) const
	{
		return ((e.y2 > _y2_mid_value) << 1) | (e.y1 > _y1_mid_value);
	}

	void create_next_level()
	{
		_nodes = new PVQuadTree [4];
		_nodes[NE].init(_y1_mid_value, _y1_max_value,
		                _y2_mid_value, _y2_max_value,
		                _max_level - 1);

		_nodes[SE].init(_y1_mid_value, _y1_max_value,
		                _y2_min_value, _y2_mid_value,
		                _max_level - 1);

		_nodes[SW].init(_y1_min_value, _y1_mid_value,
		                _y2_min_value, _y2_mid_value,
		                _max_level - 1);

		_nodes[NW].init(_y1_min_value, _y1_mid_value,
		                _y2_mid_value, _y2_max_value,
		                _max_level - 1);

		for (unsigned i = 0; i < _datas.size(); ++i) {
			PVQuadTreeEntry &e = _datas.at(i);
			_nodes[compute_index(e)]._datas.push_back(e);
		}
		_datas.clear();
	}

private:
	template <typename RESULT, typename __impl::f_traverse_dim<RESULT>::function_type F>
	struct visit_y1
	{
		static void f(PVQuadTree const& obj, uint32_t y1_min, uint32_t y1_max, RESULT &result)
		{
			if (obj._nodes != 0) {
				if (obj._y1_mid_value < y1_max) {
					f(obj._nodes[NE], y1_min, y1_max, result);
					f(obj._nodes[SE], y1_min, y1_max, result);
				}
				if (y1_min < obj._y1_mid_value) {
					f(obj._nodes[NW], y1_min, y1_max, result);
					f(obj._nodes[SW], y1_min, y1_max, result);
				}
			} else {
				F(obj._datas, result);
			}
		}
	};

	template <typename RESULT, typename __impl::f_traverse_dim<RESULT>::function_type F>
	struct visit_y2
	{
		static void f(PVQuadTree const& obj, uint32_t y2_min, uint32_t y2_max, RESULT &result)
		{
			if (obj._nodes != 0) {
				if (obj._y2_mid_value < y2_max) {
					f(obj._nodes[NE], y2_min, y2_max, result);
					f(obj._nodes[SE], y2_min, y2_max, result);
				}
				if (y2_min < obj._y2_mid_value) {
					f(obj._nodes[NW], y2_min, y2_max, result);
					f(obj._nodes[SW], y2_min, y2_max, result);
				}
			} else {
				F(obj._datas, result);
			}
		}
	};

	template <typename RESULT, typename __impl::f_traverse_dim<RESULT>::function_type F>
	struct visit_y1y2
	{
		static void f(PVQuadTree const& obj, uint32_t y1_min, uint32_t y1_max, uint32_t y2_min, uint32_t y2_max, RESULT &result)
		{
			if (obj._nodes != 0) {
				if(obj._y1_mid_value < y1_max) {
					if(obj._y2_mid_value < y2_max) {
						f(obj._nodes[NE], y1_min, y1_max, y2_min, y2_max, result);
					}
					if(y2_min < obj._y2_mid_value) {
						f(obj._nodes[SE], y1_min, y1_max, y2_min, y2_max, result);
					}
				}
				if(y1_min < obj._y1_mid_value) {
					if(obj._y2_mid_value < y2_max) {
						f(obj._nodes[NW], y1_min, y1_max, y2_min, y2_max, result);
					}
					if(y2_min < obj._y2_mid_value) {
						f(obj._nodes[SW], y1_min, y1_max, y2_min, y2_max, result);
					}
				}
			} else {
				F(obj._datas, result);
			}
		}
	};

	template <typename RESULT, typename __impl::f_traverse_sel<RESULT>::function_type F>
	struct visit_sel
	{
		static void f(PVQuadTree const& obj, const Picviz::PVSelection &selection, RESULT &result)
		{
			if(obj._nodes != 0) {
				f(obj._nodes[NE], selection, result);
				f(obj._nodes[SE], selection, result);
				f(obj._nodes[NW], selection, result);
				f(obj._nodes[SW], selection, result);
			} else {
				F(obj._datas, selection, result);
			}
		}
	};

	template <typename RESULT, typename __impl::f_traverse_sel<RESULT>::function_type F>
	struct visit_y1_sel
	{
		static void f(PVQuadTree const& obj, uint32_t y1_min, uint32_t y1_max, const Picviz::PVSelection &selection, RESULT &result)
		{
			if (obj._nodes != 0) {
				if (obj._y1_mid_value < y1_max) {
					f(obj._nodes[NE], y1_min, y1_max, selection, result);
					f(obj._nodes[SE], y1_min, y1_max, selection, result);
				}
				if (y1_min < obj._y1_mid_value) {
					f(obj._nodes[NW], y1_min, y1_max, selection, result);
					f(obj._nodes[SW], y1_min, y1_max, selection, result);
				}
			} else {
				F(obj._datas, selection, result);
			}
		}
	};

	template <typename RESULT, typename __impl::f_traverse_sel<RESULT>::function_type F>
	struct visit_y2_sel
	{
		static void f(PVQuadTree const& obj, uint32_t y2_min, uint32_t y2_max, const Picviz::PVSelection &selection, RESULT &result)
		{
			if (obj._nodes != 0) {
				if (obj._y2_mid_value < y2_max) {
					f(obj._nodes[NE], y2_min, y2_max, selection, result);
					f(obj._nodes[SE], y2_min, y2_max, selection, result);
				}
				if (y2_min < obj._y2_mid_value) {
					f(obj._nodes[NW], y2_min, y2_max, selection, result);
					f(obj._nodes[SW], y2_min, y2_max, selection, result);
				}
			} else {
				F(obj._datas, selection, result);
			}
		}
	};

	template <typename RESULT, typename __impl::f_traverse_sel<RESULT>::function_type F>
	struct visit_y1y2_sel
	{
		static void f(PVQuadTree const& obj, uint32_t y1_min, uint32_t y1_max, uint32_t y2_min, uint32_t y2_max, const Picviz::PVSelection &selection, RESULT &result)
		{
			if (obj._nodes != 0) {
				if(obj._y1_mid_value < y1_max) {
					if(obj._y2_mid_value < y2_max) {
						f(obj._nodes[NE], y1_min, y1_max, y2_min, y2_max, selection, result);
					}
					if(y2_min < obj._y2_mid_value) {
						f(obj._nodes[SE], y1_min, y1_max, y2_min, y2_max, selection, result);
					}
				}
				if(y1_min < obj._y1_mid_value) {
					if(obj._y2_mid_value < y2_max) {
						f(obj._nodes[NW], y1_min, y1_max, y2_min, y2_max, selection, result);
					}
					if(y2_min < obj._y2_mid_value) {
						f(obj._nodes[SW], y1_min, y1_max, y2_min, y2_max, selection, result);
					}
				}
			} else {
				F(obj._datas, selection, result);
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
			__impl::f_get_entry_sel(_datas, selection, new_tree._datas);
		}
	}


private:
	pvquadtree_entries_t  _datas;
	PVQuadTree           *_nodes;

	uint32_t              _y1_min_value;
	uint32_t              _y1_max_value;
	uint32_t              _y2_min_value;
	uint32_t              _y2_max_value;

	uint32_t              _y1_mid_value;
	uint32_t              _y2_mid_value;
	uint32_t              _max_level;
};

#undef SW
#undef SE
#undef NW
#undef NE

}

#endif // PARALLELVIEW_PVQUADTREE_H
