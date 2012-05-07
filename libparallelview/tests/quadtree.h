#ifndef QUADTREE_H
#define QUADTREE_H

#include <stdint.h>

#include <omp.h>

#include <stdlib.h>
#include <string.h>

#include <tbb/concurrent_vector.h>

#include "vector.h"

template <class C>
class PVconcurrentVector : public tbb::concurrent_vector<C>
{
public:
	bool is_null()
	{
		return tbb::concurrent_vector<C>::empty() && (tbb::concurrent_vector<C>::capacity() == 0);
	}
};

template <class DataContainer, class Data>
class PVQuadTree
{
#ifdef USE_CONC_VEC
	typedef PVconcurrentVector<entry> list_rows_t;
#else
	typedef DataContainer list_rows_t;
#endif

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
		_datas.reserve(MAX_SIZE + 1);
		_nodes[0] = _nodes[1] = _nodes[2] = _nodes[3] = 0;
	}

	PVQuadTree(uint32_t y1_mid_value, uint32_t y2_mid_value, int max_level, int cur_level) :
		_y1_mid_value(y1_mid_value),
		_y2_mid_value(y2_mid_value),
		_max_level(max_level),
		_cur_level(cur_level)
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

	void insert(const entry &e) {
		// searching for the right child
		PVQuadTree *qt = this;
		while (qt->_datas.is_null()) {
			qt = qt->_nodes[qt->compute_index(e)];
		}

		// insertion
		qt->_datas.push_back(e);

		// does the current node must be splitted?
		if((qt->_datas.size() >= MAX_SIZE) && (qt->_cur_level < qt->_max_level)) {
			qt->create_next_level();
		}
	}

	void get_y1_first(int vmin, int vmax, Data &result)
	{
		if(_datas.is_null()) {
			if(_y1_mid_value < vmax) {
				_nodes[NW].get_y1_first(vmin, vmax, result);
				_nodes[SW].get_y1_first(vmin, vmax, result);
			}
			if(vmin < _y1_mid_value) {
				_nodes[NE].get_y1_first(vmin, vmax, result);
				_nodes[SE].get_y1_first(vmin, vmax, result);
			}
		} else {
			Data current = _datas.back();
			if(result.idx <= current.idx) {
				result = current;
			}
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

#ifdef USE_CONC_VEC
#pragma omp parallel for
		for(unsigned i = 0; i < _datas.size(); ++i) {
			entry &e = _datas[i];
			_nodes[compute_index(e)]->_datas.push_back(e);
		}
#else
		for(unsigned i = 0; i < _datas.size(); ++i) {
			entry &e = _datas.at(i);
			_nodes[compute_index(e)]->_datas.push_back(e);
		}
#endif
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
	uint32_t     _cur_level;
};

#endif // QUADTREE_H
