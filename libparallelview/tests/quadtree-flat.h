#ifndef QUADTREE_FLAT_H
#define QUADTREE_FLAT_H

#include <stdlib.h>

#include "quadtree.h"

// TODO: replace use *{min,max}_value by a precomputation step

template <class DataContainer, class Data>
class PVQuadTreeFlatBase
{
public:
	PVQuadTreeFlatBase() {}

	void set(uint32_t y1_min_value, uint32_t y1_max_value, uint32_t y2_min_value, uint32_t y2_max_value, uint32_t position, int max_level)
	{
		_y1_min_value = y1_min_value;
		_y1_max_value = y1_max_value;
		_y2_min_value = y2_min_value;
		_y2_max_value = y2_max_value;
		_position = position;
		_max_level = max_level;
		_y1_mid_value = (_y1_min_value + _y1_max_value) / 2;
		_y2_mid_value = (_y2_min_value + _y2_max_value) / 2;
		_datas.reserve(MAX_SIZE + 1);
	}

	inline size_t memory() const
	{
		return sizeof(PVQuadTreeFlatBase<DataContainer, Data>) - sizeof(DataContainer) + _datas.memory();
	}

	uint32_t children()
	{
		return (_position << 2) + 1;
	}

	int compute_index(const entry &e) const
	{
		return ((e.y2 > _y2_mid_value) << 1) | (e.y1 > _y1_mid_value);
	}

	void create_next_level(PVQuadTreeFlatBase *tab)
	{
		unsigned pos = children();

		tab[NE].set(_y1_mid_value, _y1_max_value,
		            _y2_mid_value, _y2_max_value,
		            pos + NE,
		            _max_level - 1);

		tab[SE].set(_y1_mid_value, _y1_max_value,
		            _y2_min_value, _y2_mid_value,
		            pos + SE,
		            _max_level - 1);

		tab[SW].set(_y1_min_value, _y1_mid_value,
		            _y2_min_value, _y2_mid_value,
		            pos + SW,
		            _max_level - 1);

		tab[NW].set(_y1_min_value, _y1_mid_value,
		            _y2_mid_value, _y2_max_value,
		            pos + NW,
		            _max_level - 1);

		for(unsigned i = 0; i < _datas.size(); ++i) {
			entry &e = _datas.at(i);
			tab[compute_index(e)]._datas.push_back(e);
		}
		_datas.clear();
	}

	bool compare(PVQuadTree<DataContainer, Data> &qt, PVQuadTreeFlatBase* trees)
	{
		if(_datas.is_null() && qt._datas.is_null()) {
			// *this sont des noeuds, on va voir les
			// noeuds fils
			unsigned p = children();
			return trees[p].compare(*qt._nodes[0], trees)
				&& trees[p+1].compare(*qt._nodes[1], trees)
				&& trees[p+2].compare(*qt._nodes[2], trees)
				&& trees[p+3].compare(*qt._nodes[3], trees);
		} else if(_datas.is_null()) {
			// *this et qt ne sont pas de même type
			return false;
		} else if(_datas.size() != qt._datas.size()) {
			// *this et qt sont de même type mais
			// pas avec le même nombre d'éléments
			return false;
		} else {
			// même type et même nombre d'éléments
			for(unsigned i = 0; i < _datas.size(); ++i) {
				if(are_diff(_datas.at(i), qt._datas.at(i))) {
					return false;
				}
			}
			return true;
		}
	}

public:
	DataContainer     _datas;
	unsigned _nodes[4];

       	uint32_t _y1_min_value;
	uint32_t _y1_max_value;
	uint32_t _y2_min_value;
	uint32_t _y2_max_value;

	uint32_t _y1_mid_value;
	uint32_t _y2_mid_value;
	uint32_t _position;
	uint32_t _max_level;
};

template <class DataContainer, class Data>
class PVQuadTreeFlat
{
public:
	PVQuadTreeFlat(uint32_t y1_min_value, uint32_t y1_max_value, uint32_t y2_min_value, uint32_t y2_max_value, int max_level)
	{
		_count = 1 << (max_level * 2);
		_trees = (PVQuadTreeFlatBase<DataContainer, Data>*)calloc(_count, sizeof(PVQuadTreeFlatBase<DataContainer, Data>));
		_trees[0].set(y1_min_value, y1_max_value, y2_min_value, y2_max_value, 0, max_level);
	}

	inline size_t memory() const
	{
		size_t mem = _count * (sizeof (PVQuadTreeFlatBase<DataContainer, Data>) - sizeof(DataContainer));
		for(unsigned i = 0; i < _count; ++i) {
			mem += _trees[i].memory();
		}
		return mem;
	}

	void insert(const entry &e) {
		// searching for the right child
		PVQuadTreeFlatBase<DataContainer, Data> *qt = &_trees[0];
		while (qt->_datas.is_null()) {
			qt = &_trees[qt->children() + qt->compute_index(e)];
		}

		// insertion
		qt->_datas.push_back(e);

		// does the current node must be splitted?
		if((qt->_datas.size() >= MAX_SIZE) && (qt->_max_level)) {
			qt ->create_next_level(&_trees[qt->children()]);
		}
	}

	bool compare(PVQuadTree<DataContainer, Data> &qt)
	{
		return _trees[0].compare(qt, _trees);
	}

private:
	unsigned                  _count;
	PVQuadTreeFlatBase<DataContainer, Data> *_trees;
};

#endif // QUADTREE_FLAT_H
