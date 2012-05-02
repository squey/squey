#ifndef QUADTREE_H
#define QUADTREE_H

#include <vector>
#include <algorithm>
#include <stdint.h>

#include <omp.h>

#include <stdlib.h>
#include <string.h>

typedef uint32_t offset;

struct entry {
	offset y1, y2;
	uint32_t idx;
};

enum {
	SW = 0,
	SE,
	NW,
	NE
};

#define MAX_SIZE 10000
//#define MAX_SIZE 3
#define NBITS_INDEX 10

namespace Picviz
{
template <class C>
class PVVector
{
public:
	PVVector(unsigned size = 1000, unsigned increment = 1000) :
		_size(size),
		_increment(increment),
		_index(0)
	{
		_array = 0;
	}

	~PVVector()
	{
		if (_array) {
			free(_array);
		}
	}

	void reserve(unsigned size)
	{
		reallocate(size);
	}

	unsigned size()
	{
		return _index;
	}

	void clear()
	{
		_index = 0;
	}

	C &at(int i)
	{
		return _array[i];
	}

	void push_back(C &c)
	{
		if (_index == _size) {
			reallocate();
		}
		_array[_index++] = c;
	}

private:
	void reallocate(unsigned size = 0)
	{
		if (size) {
			_array = (C*) realloc(_array, (size) * sizeof(C));
			_size = size;
		} else {
			_array = (C*) realloc(_array, (_size + _increment) * sizeof(C));
			_size += _increment;
		}
	}

private:
	C        *_array;
	unsigned  _size;
	unsigned  _increment;
	unsigned  _index;
};
}

class PVQuadTree
{
	typedef std::vector<entry> list_rows_t;
	// typedef Picviz::PVVector<entry> list_rows_t;

public:
	PVQuadTree(uint32_t y1_min_value, uint32_t y1_max_value, uint32_t y2_min_value, uint32_t y2_max_value, int max_level, int cur_level = 0) :
		_y1_min_value(y1_min_value),
		_y1_max_value(y1_max_value),
		_y2_min_value(y2_min_value),
		_y2_max_value(y2_max_value),
		_max_level(max_level),
		_cur_level(cur_level)
	{
		_y1_mid_value = (_y1_min_value + _y1_max_value) / 2;
		_y2_mid_value = (_y2_min_value + _y2_max_value) / 2;
		_datas = new list_rows_t();
		_datas->reserve(MAX_SIZE + 1);
		_nodes[0] = _nodes[1] = _nodes[2] = _nodes[3] = 0;
		_first.idx = 0;
	}

	PVQuadTree(uint32_t y1_mid_value, uint32_t y2_mid_value, int max_level, int cur_level) :
		_y1_mid_value(y1_mid_value),
		_y2_mid_value(y2_mid_value),
		_max_level(max_level),
		_cur_level(cur_level)
	{
		_datas = new list_rows_t();
		_datas->reserve(MAX_SIZE + 1);
		_nodes[0] = _nodes[1] = _nodes[2] = _nodes[3] = 0;
		_first.idx = 0;
	}

	~PVQuadTree() {
		if(_datas != 0) {
			delete _datas;
		} else {
			for(int i = 0; i < 4; ++i) {
				delete _nodes[i];
			}
		}
	}

	void insert(entry &e) {
		if (e.idx >= _first.idx) {
			_first = e;
		}

		// searching for the right child
		PVQuadTree *qt = this;
		while (qt->_datas == 0) {
			qt = qt->_nodes[qt->compute_idx(e)];
		}

		// insertion
		qt->_datas->push_back(e);

		// does the current node must be splitted?
		if((qt->_datas->size() >= MAX_SIZE) && (qt->_cur_level < qt->_max_level)) {
			qt->create_next_level();
		}
	}

	void dump(unsigned indent = 0)
	{
		if (_datas) {
			for(unsigned i = 0; i < indent; ++i) {
				std::cout << "  ";
			}
			for(unsigned i = 0; i < _datas->size(); ++i) {
				entry &e = _datas->at(i);
				std::cout << "(" << e.y1 << ", " << e.y2 << ", " << e.idx << ") ";
			}
			std::cout << std::endl;
		} else {
			for(unsigned i = 0; i < indent; ++i) {
				std::cout << "  ";
			}
			std::cout << "NE" << std::endl;
			_nodes[NE]->dump(indent + 2);
			for(unsigned i = 0; i < indent; ++i) {
				std::cout << "  ";
			}
			std::cout << "SE" << std::endl;
			_nodes[SE]->dump(indent + 2);
			for(unsigned i = 0; i < indent; ++i) {
				std::cout << "  ";
			}
			std::cout << "NW" << std::endl;
			_nodes[NW]->dump(indent + 2);
			for(unsigned i = 0; i < indent; ++i) {
				std::cout << "  ";
			}
			std::cout << "SW" << std::endl;
			_nodes[SW]->dump(indent + 2);
		}
	}

	entry get_first()
	{
		return _first;
	}

private:
	int compute_idx(entry &e)
	{
		return ((e.y2 > _y2_mid_value) << 1) | (e.y1 > _y1_mid_value);
	}

	void create_next_level()
	{
#ifdef AA
		// les calculs ci-dessous font perdre 60-70 ms sur 10.000.000 insertions
		unsigned y1d = _y1_mid_value >> _cur_level;
		unsigned y2d = _y2_mid_value >> _cur_level;
		unsigned y1_1 = _y1_mid_value + y1d;
		unsigned y2_1 = _y2_mid_value + y2d;
		unsigned y1_0 = _y1_mid_value - y1d;
		unsigned y2_0 = _y2_mid_value - y2d;


		_nodes[NE] = new PVQuadTree(y1_1,
		                            y2_1,
		                            _max_level, _cur_level + 1);

		_nodes[SE] = new PVQuadTree(y1_1,
		                            y2_0,
		                            _max_level, _cur_level + 1);

		_nodes[SW] = new PVQuadTree(y1_0,
		                            y2_0,
		                            _max_level, _cur_level + 1);

		_nodes[NW] = new PVQuadTree(y1_0,
		                            y2_1,
		                            _max_level, _cur_level + 1);
#else
		_nodes[NE] = new PVQuadTree(_y1_mid_value, _y1_max_value,
		                            _y2_mid_value, _y2_max_value,
		                            _max_level, _cur_level + 1);

		_nodes[SE] = new PVQuadTree(_y1_mid_value, _y1_max_value,
		                            _y2_min_value, _y2_mid_value,
		                            _max_level, _cur_level + 1);

		_nodes[SW] = new PVQuadTree(_y1_min_value, _y1_mid_value,
		                            _y2_min_value, _y2_mid_value,
		                            _max_level, _cur_level + 1);

		_nodes[NW] = new PVQuadTree(_y1_min_value, _y1_mid_value,
		                            _y2_mid_value, _y2_max_value,
		                            _max_level, _cur_level + 1);
#endif

#ifdef RH
#pragma omp parallel
		for(unsigned i = 0; i < _datas->size(); ++i) {
			entry &e = (*_datas)[i];
			int idx = compute_idx(e);
			if (idx == omp_get_thread_num()) {
				_nodes[idx]->_datas->push_back(e);
			}
		}
#else
		for(unsigned i = 0; i < _datas->size(); ++i) {
			entry &e = _datas->at(i);
			_nodes[compute_idx(e)]->_datas->push_back(e);
		}
#endif
		delete _datas;
		_datas = 0;
	}

private:
	list_rows_t *_datas;
	PVQuadTree  *_nodes[4];
	entry        _first;

	uint32_t     _y1_min_value;
	uint32_t     _y1_max_value;
	uint32_t     _y2_min_value;
	uint32_t     _y2_max_value;

	uint32_t     _y1_mid_value;
	uint32_t     _y2_mid_value;
	uint32_t     _max_level;
	uint32_t     _cur_level;
};

#endif // QUADTREE_H
