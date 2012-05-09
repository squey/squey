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
				_nodes[NW]->extract_first_y1(y1_min, y1_max, results);
				_nodes[SW]->extract_first_y1(y1_min, y1_max, results);
			}
			if(y1_min < _y1_mid_value) {
				_nodes[NE]->extract_first_y1(y1_min, y1_max, results);
				_nodes[SE]->extract_first_y1(y1_min, y1_max, results);
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
					_nodes[NW]->extract_first_y1y2(y1_min, y1_max, y2_min, y2_max, results);
				}
				if(y2_min < _y2_mid_value) {
					_nodes[SW]->extract_first_y1y2(y1_min, y1_max, y2_min, y2_max, results);
				}
			}
			if(y1_min < _y1_mid_value) {
				if(_y2_mid_value < y2_max) {
					_nodes[NE]->extract_first_y1y2(y1_min, y1_max, y2_min, y2_max, results);
				}
				if(y2_min < _y2_mid_value) {
					_nodes[SE]->extract_first_y1y2(y1_min, y1_max, y2_min, y2_max, results);
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
				_nodes[NW]->extract_first_y1_bci(y1_min, y1_max, results);
				_nodes[SW]->extract_first_y1_bci(y1_min, y1_max, results);
			}
			if(y1_min < _y1_mid_value) {
				_nodes[NE]->extract_first_y1_bci(y1_min, y1_max, results);
				_nodes[SE]->extract_first_y1_bci(y1_min, y1_max, results);
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
					_nodes[NW]->extract_first_y1y2_bci(y1_min, y1_max, y2_min, y2_max, results);
				}
				if(y2_min < _y2_mid_value) {
					_nodes[SW]->extract_first_y1y2_bci(y1_min, y1_max, y2_min, y2_max, results);
				}
			}
			if(y1_min < _y1_mid_value) {
				if(_y2_mid_value < y2_max) {
					_nodes[NE]->extract_first_y1y2_bci(y1_min, y1_max, y2_min, y2_max, results);
				}
				if(y2_min < _y2_mid_value) {
					_nodes[SE]->extract_first_y1y2_bci(y1_min, y1_max, y2_min, y2_max, results);
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

private:
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
