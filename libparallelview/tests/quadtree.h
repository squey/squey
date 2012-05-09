#ifndef QUADTREE_H
#define QUADTREE_H

#include <stdint.h>

#include <stdlib.h>
#include <string.h>

#include "vector.h"

#include <pvparallelview/PVBCICode.h>

template <class DataContainer, class Data>
class PVQuadTree
{
	typedef DataContainer list_rows_t;

public:
	PVQuadTree(uint32_t y1_min_value, uint32_t y1_max_value, uint32_t y2_min_value, uint32_t y2_max_value, int max_level) :
		_y1_min_value(y1_min_value),
		_y1_max_value(y1_max_value),
		_y2_min_value(y2_min_value),
		_y2_max_value(y2_max_value),
		_max_level(max_level)
	{
		_y1_mid_value = (_y1_min_value + _y1_max_value) / 2;
		_y2_mid_value = (_y2_min_value + _y2_max_value) / 2;
		_datas.reserve(MAX_SIZE + 1);
		_nodes[0] = _nodes[1] = _nodes[2] = _nodes[3] = 0;
	}

	PVQuadTree(uint32_t y1_mid_value, uint32_t y2_mid_value, int max_level) :
		_y1_mid_value(y1_mid_value),
		_y2_mid_value(y2_mid_value),
		_max_level(max_level)
	{
		_datas.reserve(MAX_SIZE + 1);
		_nodes[0] = _nodes[1] = _nodes[2] = _nodes[3] = 0;
	}

	~PVQuadTree() {
		if(_datas.is_null() == false) {
			_datas.clear();
		} else {
			for(int i = 0; i < 4; ++i) {
				delete _nodes[i];
			}
		}
	}

	inline size_t memory() const
	{
		size_t mem = sizeof (PVQuadTree<DataContainer, Data>) - sizeof(DataContainer) + _datas.memory();
		if(_nodes[0] != 0) {
			mem += _nodes[0]->memory();
			mem += _nodes[1]->memory();
			mem += _nodes[2]->memory();
			mem += _nodes[3]->memory();
		}
		return mem;
	}

	unsigned elements() const
	{
		if(_datas.is_null()) {
			return _nodes[0]->elements() + _nodes[1]->elements() + _nodes[2]->elements() +_nodes[3]->elements();
		} else {
			return _datas.size();
		}
	}

	void dump(std::ostream &os) const
	{
		if(_datas.is_null()) {
			for(unsigned i = 0; i < 4; ++i) {
				_nodes[i]->dump(os);
			}
		} else {
			for(unsigned i = 0; i < _datas.size(); ++i) {
				entry e = _datas.at(i);
				os << e.y1 << " " << e.y2 << std::endl;
			}
		}
	}

	void insert(const entry &e) {
		// searching for the right child
		PVQuadTree *qt = this;
		while (qt->_datas.is_null()) {
			qt = qt->_nodes[qt->compute_index(e)];
		}

		// insertion
		qt->_datas.push_back(e);

		// does the current node must be splitted?
		if((qt->_datas.size() >= MAX_SIZE) && qt->_max_level) {
			qt->create_next_level();
		}
	}

	void extract_first_y1(uint32_t y1_min, uint32_t y1_max, std::vector<Data> &results) const
	{
		if(_datas.is_null()) {
			if(_y1_mid_value < y1_max) {
				_nodes[NE]->extract_first_y1(y1_min, y1_max, results);
				_nodes[SE]->extract_first_y1(y1_min, y1_max, results);
			}
			if(y1_min < _y1_mid_value) {
				_nodes[NW]->extract_first_y1(y1_min, y1_max, results);
				_nodes[SW]->extract_first_y1(y1_min, y1_max, results);
			}
		} else if(_datas.size() != 0) {
			results.push_back(_datas.at(0));
		}
	}

	void extract_first_y2(uint32_t y2_min, uint32_t y2_max, std::vector<Data> &results) const
	{
		if(_datas.is_null()) {
			if(_y2_mid_value < y2_max) {
				_nodes[NW]->extract_first_y2(y2_min, y2_max, results);
				_nodes[NE]->extract_first_y2(y2_min, y2_max, results);
			}
			if(y2_min < _y2_mid_value) {
				_nodes[SW]->extract_first_y2(y2_min, y2_max, results);
				_nodes[SE]->extract_first_y2(y2_min, y2_max, results);
			}
		} else if(_datas.size() != 0) {
			results.push_back(_datas.at(0));
		}
	}

	void extract_first_y1y2(uint32_t y1_min, uint32_t y1_max, uint32_t y2_min, uint32_t y2_max, std::vector<Data> &results) const
	{
		if(_datas.is_null()) {
			if(_y1_mid_value < y1_max) {
				if(_y2_mid_value < y2_max) {
					_nodes[NE]->extract_first_y1y2(y1_min, y1_max, y2_min, y2_max, results);
				}
				if(y2_min < _y2_mid_value) {
					_nodes[SE]->extract_first_y1y2(y1_min, y1_max, y2_min, y2_max, results);
				}
			}
			if(y1_min < _y1_mid_value) {
				if(_y2_mid_value < y2_max) {
					_nodes[NW]->extract_first_y1y2(y1_min, y1_max, y2_min, y2_max, results);
				}
				if(y2_min < _y2_mid_value) {
					_nodes[SW]->extract_first_y1y2(y1_min, y1_max, y2_min, y2_max, results);
				}
			}
		} else if(_datas.size() != 0) {
			results.push_back(_datas.at(0));
		}
	}

	void extract_first_y1_bci(uint32_t y1_min, uint32_t y1_max, std::vector<PVParallelView::PVBCICode> &results) const
	{
		if(_datas.is_null()) {
			if(_y1_mid_value < y1_max) {
				_nodes[NE]->extract_first_y1_bci(y1_min, y1_max, results);
				_nodes[SE]->extract_first_y1_bci(y1_min, y1_max, results);
			}
			if(y1_min < _y1_mid_value) {
				_nodes[NW]->extract_first_y1_bci(y1_min, y1_max, results);
				_nodes[SW]->extract_first_y1_bci(y1_min, y1_max, results);
			}
		} else if(_datas.size() != 0) {
			entry e = _datas.at(0);
			PVParallelView::PVBCICode code;
			code.s.idx = e.idx;
			code.s.l = e.y1 >> 22;
			code.s.r = e.y2 >> 22;
			code.s.color = random() & 255;
			results.push_back(code);
		}
	}

	void extract_first_y2_bci(uint32_t y2_min, uint32_t y2_max, std::vector<PVParallelView::PVBCICode> &results) const
	{
		if(_datas.is_null()) {
			if(_y2_mid_value < y2_max) {
				_nodes[NW]->extract_first_y2_bci(y2_min, y2_max, results);
				_nodes[NE]->extract_first_y2_bci(y2_min, y2_max, results);
			}
			if(y2_min < _y2_mid_value) {
				_nodes[SW]->extract_first_y2_bci(y2_min, y2_max, results);
				_nodes[SE]->extract_first_y2_bci(y2_min, y2_max, results);
			}
		} else if(_datas.size() != 0) {
			entry e = _datas.at(0);
			PVParallelView::PVBCICode code;
			code.s.idx = e.idx;
			code.s.l = e.y1 >> 22;
			code.s.r = e.y2 >> 22;
			code.s.color = random() & 255;
			results.push_back(code);
		}
	}

	void extract_first_y1y2_bci(uint32_t y1_min, uint32_t y1_max, uint32_t y2_min, uint32_t y2_max, std::vector<PVParallelView::PVBCICode> &results) const
	{
		if(_datas.is_null()) {
			if(_y1_mid_value < y1_max) {
				if(_y2_mid_value < y2_max) {
					_nodes[NE]->extract_first_y1y2_bci(y1_min, y1_max, y2_min, y2_max, results);
				}
				if(y2_min < _y2_mid_value) {
					_nodes[SE]->extract_first_y1y2_bci(y1_min, y1_max, y2_min, y2_max, results);
				}
			}
			if(y1_min < _y1_mid_value) {
				if(_y2_mid_value < y2_max) {
					_nodes[NW]->extract_first_y1y2_bci(y1_min, y1_max, y2_min, y2_max, results);
				}
				if(y2_min < _y2_mid_value) {
					_nodes[SW]->extract_first_y1y2_bci(y1_min, y1_max, y2_min, y2_max, results);
				}
			}
		} else if(_datas.size() != 0) {
			entry e = _datas.at(0);
			PVParallelView::PVBCICode code;
			code.s.idx = e.idx;
			code.s.l = e.y1 >> 22;
			code.s.r = e.y2 >> 22;
			code.s.color = random() & 255;
			results.push_back(code);
		}
	}

	PVQuadTree<DataContainer, Data> *extract_subtree_y1(uint32_t y1_min, uint32_t y1_max) const
	{
		PVQuadTree<DataContainer, Data> *new_tree = new PVQuadTree<DataContainer, Data>(*this);
		if(_datas.is_null()) {
			if(_y1_mid_value < y1_max) {
				new_tree->_nodes[NE] = _nodes[NW]->extract_subtree_y1(y1_min, y1_max);
				new_tree->_nodes[SE] = _nodes[SW]->extract_subtree_y1(y1_min, y1_max);
			} else {
				new_tree->_nodes[NE] = new PVQuadTree<DataContainer, Data>(*_nodes[NE]);
				new_tree->_nodes[NE]->_datas.reserve(1);
				new_tree->_nodes[SE] = new PVQuadTree<DataContainer, Data>(*_nodes[SE]);
				new_tree->_nodes[SE]->_datas.reserve(1);
			}
			if(y1_min < _y1_mid_value) {
				new_tree->_nodes[NW] = _nodes[NW]->extract_subtree_y1(y1_min, y1_max);
				new_tree->_nodes[SW] = _nodes[SW]->extract_subtree_y1(y1_min, y1_max);
			} else {
				new_tree->_nodes[NW] = new PVQuadTree<DataContainer, Data>(*_nodes[NW]);
				new_tree->_nodes[NW]->_datas.reserve(1);
				new_tree->_nodes[SW] = new PVQuadTree<DataContainer, Data>(*_nodes[SW]);
				new_tree->_nodes[SW]->_datas.reserve(1);
			}
		} else if(_datas.size() != 0) {
			new_tree->_datas = _datas;
		} else {
			new_tree->_datas.reserve(1);
		}
		return new_tree;
	}

	PVQuadTree<DataContainer, Data> *extract_subtree_y2(uint32_t y2_min, uint32_t y2_max) const
	{
		PVQuadTree<DataContainer, Data> *new_tree = new PVQuadTree<DataContainer, Data>(*this);
		if(_datas.is_null()) {
			if(_y2_mid_value < y2_max) {
				new_tree->_nodes[NW] = _nodes[NW]->extract_subtree_y2(y2_min, y2_max);
				new_tree->_nodes[NE] = _nodes[NE]->extract_subtree_y2(y2_min, y2_max);
			} else {
				new_tree->_nodes[NW] = new PVQuadTree<DataContainer, Data>(*_nodes[NW]);
				new_tree->_nodes[NW]->_datas.reserve(1);
				new_tree->_nodes[NE] = new PVQuadTree<DataContainer, Data>(*_nodes[NE]);
				new_tree->_nodes[NE]->_datas.reserve(1);
			}
			if(y2_min < _y2_mid_value) {
				new_tree->_nodes[SW] = _nodes[SW]->extract_subtree_y2(y2_min, y2_max);
				new_tree->_nodes[SE] = _nodes[SE]->extract_subtree_y2(y2_min, y2_max);
			} else {
				new_tree->_nodes[SW] = new PVQuadTree<DataContainer, Data>(*_nodes[SW]);
				new_tree->_nodes[SW]->_datas.reserve(1);
				new_tree->_nodes[SE] = new PVQuadTree<DataContainer, Data>(*_nodes[SE]);
				new_tree->_nodes[SE]->_datas.reserve(1);
			}
		} else if(_datas.size() != 0) {
			new_tree->_datas = _datas;
		} else {
			new_tree->_datas.reserve(1);
		}
		return new_tree;
	}

	PVQuadTree<DataContainer, Data> *extract_subtree_y1y2(uint32_t y1_min, uint32_t y1_max, uint32_t y2_min, uint32_t y2_max) const
	{
		PVQuadTree<DataContainer, Data> *new_tree = new PVQuadTree<DataContainer, Data>(*this);
		if(_datas.is_null()) {
			if(_y1_mid_value < y1_max) {
				if(_y2_mid_value < y2_max) {
					new_tree->_nodes[NE] = _nodes[NE]->extract_subtree_y1y2(y1_min, y1_max, y2_min, y2_max);
				} else {
					new_tree->_nodes[NE] = new PVQuadTree<DataContainer, Data>(*_nodes[NE]);
					new_tree->_nodes[NE]->_datas.reserve(1);
				}
				if(y2_min < _y2_mid_value) {
					new_tree->_nodes[SE] = _nodes[SE]->extract_subtree_y1y2(y1_min, y1_max, y2_min, y2_max);
				} else {
					new_tree->_nodes[SE] = new PVQuadTree<DataContainer, Data>(*_nodes[SE]);
					new_tree->_nodes[SE]->_datas.reserve(1);
				}
			} else {
				new_tree->_nodes[NE] = new PVQuadTree<DataContainer, Data>(*_nodes[NE]);
				new_tree->_nodes[NE]->_datas.reserve(1);
				new_tree->_nodes[SE] = new PVQuadTree<DataContainer, Data>(*_nodes[SE]);
				new_tree->_nodes[SE]->_datas.reserve(1);
			}
			if(y1_min < _y1_mid_value) {
				if(_y2_mid_value < y2_max) {
					new_tree->_nodes[NW] = _nodes[NW]->extract_subtree_y1y2(y1_min, y1_max, y2_min, y2_max);
				} else {
					new_tree->_nodes[NW] = new PVQuadTree<DataContainer, Data>(*_nodes[NW]);
					new_tree->_nodes[NW]->_datas.reserve(1);
				}
				if(y2_min < _y2_mid_value) {
					new_tree->_nodes[SW] = _nodes[SW]->extract_subtree_y1y2(y1_min, y1_max, y2_min, y2_max);
				} else {
					new_tree->_nodes[SW] = new PVQuadTree<DataContainer, Data>(*_nodes[SW]);
					new_tree->_nodes[SW]->_datas.reserve(1);
				}
			} else {
				new_tree->_nodes[NW] = new PVQuadTree<DataContainer, Data>(*_nodes[NW]);
				new_tree->_nodes[NW]->_datas.reserve(1);
				new_tree->_nodes[SW] = new PVQuadTree<DataContainer, Data>(*_nodes[SW]);
				new_tree->_nodes[SW]->_datas.reserve(1);
			}
		} else if(_datas.size() != 0) {
			new_tree->_datas = _datas;
		} else {
			new_tree->_datas.reserve(1);
		}
		return new_tree;
	}

	bool operator==(const PVQuadTree<DataContainer, Data> &qt) const
	{
		if(_datas.is_null()) {
			for(unsigned i = 0; i < 4; ++i) {
				if ((_nodes[i] == 0) || (qt._nodes[i] == 0)) {
					return false;
				}
			}
			return (*_nodes[0] == *qt._nodes[0]
			        &&
			        *_nodes[1] == *qt._nodes[1]
			        &&
			        *_nodes[2] == *qt._nodes[2]
			        &&
			        *_nodes[3] == *qt._nodes[3]);
		} else {
			return (_datas == qt._datas);
		}
	}

private:
	PVQuadTree(const PVQuadTree<DataContainer, Data> &qt)
	{
		_y1_min_value = qt._y1_min_value;
		_y1_max_value = qt._y1_max_value;
		_y2_min_value = qt._y2_min_value;
		_y2_max_value = qt._y2_max_value;
		_y1_mid_value = qt._y1_mid_value;
		_y2_mid_value = qt._y2_mid_value;
		_max_level = qt._max_level;
		_nodes[0] = _nodes[1] = _nodes[2] = _nodes[3] = 0;
	}

	int compute_index(const entry &e) const
	{
		return ((e.y2 > _y2_mid_value) << 1) | (e.y1 > _y1_mid_value);
	}

	void create_next_level()
	{
		_nodes[NE] = new PVQuadTree(_y1_mid_value, _y1_max_value,
		                            _y2_mid_value, _y2_max_value,
		                            _max_level - 1);

		_nodes[SE] = new PVQuadTree(_y1_mid_value, _y1_max_value,
		                            _y2_min_value, _y2_mid_value,
		                            _max_level - 1);

		_nodes[SW] = new PVQuadTree(_y1_min_value, _y1_mid_value,
		                            _y2_min_value, _y2_mid_value,
		                            _max_level - 1);

		_nodes[NW] = new PVQuadTree(_y1_min_value, _y1_mid_value,
		                            _y2_mid_value, _y2_max_value,
		                            _max_level - 1);

		for(unsigned i = 0; i < _datas.size(); ++i) {
			entry &e = _datas.at(i);
			_nodes[compute_index(e)]->_datas.push_back(e);
		}
		_datas.clear();
	}

public:
	list_rows_t  _datas;
	PVQuadTree  *_nodes[4];

	uint32_t     _y1_min_value;
	uint32_t     _y1_max_value;
	uint32_t     _y2_min_value;
	uint32_t     _y2_max_value;

	uint32_t     _y1_mid_value;
	uint32_t     _y2_mid_value;
	uint32_t     _max_level;
};

#endif // QUADTREE_H
