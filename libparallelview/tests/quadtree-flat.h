#ifndef QUADTREE_FLAT_H
#define QUADTREE_FLAT_H

#include <stdlib.h>

template <class Data>
class PVQuadTreeFlatBase
{
public:
	PVQuadTreeFlatBase() {}

	void set(uint32_t y1_min_value, uint32_t y1_max_value, uint32_t y2_min_value, uint32_t y2_max_value, uint32_t position, int max_level, int cur_level = 0)
	{
		_y1_min_value = y1_min_value;
		_y1_max_value = y1_max_value;
		_y2_min_value = y2_min_value;
		_y2_max_value = y2_max_value;
		_position = position;
		_max_level = max_level;
		_cur_level = cur_level;
		_y1_mid_value = (_y1_min_value + _y1_max_value) / 2;
		_y2_mid_value = (_y2_min_value + _y2_max_value) / 2;
		_datas = Data();
		_datas.reserve(MAX_SIZE + 1);
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
		            _max_level, _cur_level + 1);

		tab[SE].set(_y1_mid_value, _y1_max_value,
		            _y2_min_value, _y2_mid_value,
		            pos + SE,
		            _max_level, _cur_level + 1);

		tab[SW].set(_y1_min_value, _y1_mid_value,
		            _y2_min_value, _y2_mid_value,
		            pos + SW,
		            _max_level, _cur_level + 1);

		tab[NW].set(_y1_min_value, _y1_mid_value,
		            _y2_mid_value, _y2_max_value,
		            pos + NW,
		            _max_level, _cur_level + 1);

		for(unsigned i = 0; i < _datas.size(); ++i) {
			entry &e = _datas.at(i);
			tab[compute_index(e)]._datas.push_back(e);
		}
		_datas.clear();
	}

public:
	Data     _datas;
	unsigned _nodes[4];

       	uint32_t _y1_min_value;
	uint32_t _y1_max_value;
	uint32_t _y2_min_value;
	uint32_t _y2_max_value;

	uint32_t _y1_mid_value;
	uint32_t _y2_mid_value;
	uint32_t _position;
	uint32_t _max_level;
	uint32_t _cur_level;
};

template <class Data>
class PVQuadTreeFlat
{
public:
	PVQuadTreeFlat(uint32_t y1_min_value, uint32_t y1_max_value, uint32_t y2_min_value, uint32_t y2_max_value, int max_level)
	{
		unsigned count = 1 << (max_level + 1);
		unsigned dsize = sizeof(PVQuadTreeFlatBase<Data>);
		_trees = (PVQuadTreeFlatBase<Data>*)calloc(count, dsize);
		//_trees = new PVQuadTreeFlatBase<Data> [count];
		_trees[0].set(y1_min_value, y1_max_value, y2_min_value, y2_max_value, 0, max_level, 0);
	}

	void insert(const entry &e) {
		// searching for the right child
		PVQuadTreeFlatBase<Data> *qt = &_trees[0];
		while (qt->_datas.is_null()) {
			qt = &_trees[qt->children() + qt->compute_index(e)];
		}

		// insertion
		qt->_datas.push_back(e);

		// does the current node must be splitted?
		if((qt->_datas.size() >= MAX_SIZE) && (qt->_cur_level < qt->_max_level)) {
			qt ->create_next_level(&_trees[qt->children()]);
		}
	}

private:
	PVQuadTreeFlatBase<Data> *_trees;
};

#endif // QUADTREE_FLAT_H
