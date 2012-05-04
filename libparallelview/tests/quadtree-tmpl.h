#ifndef QUADTREE_TMPL_H
#define QUADTREE_TMPL_H

#include <stdint.h>

#include <stdlib.h>
#include <string.h>

template <class Data>
class PVQuadTreeTmplBase
{
public:
	PVQuadTreeTmplBase() :
		_max_level(0),
		_cur_level(0)
	{
	}

	PVQuadTreeTmplBase(int max_level, int cur_level = 0) :
		_max_level(max_level),
		_cur_level(cur_level)
	{
		_datas.reserve(MAX_SIZE + 1);
	}

	void set(int max_level, int cur_level)
	{
		_max_level = max_level;
		_cur_level = cur_level;
		_datas.reserve(MAX_SIZE + 1);
	}

	void insert_(entry &e)
	{
		if(_datas.is_null()) {
			do_sub_insert(e);
		} else {
			_datas.push_back(e);
			if((_datas.size() >= MAX_SIZE) && (_cur_level < _max_level)) {
				create_next_level();
			}
		}
	}

	void insert(const entry &e) {
		// searching for the right child
		PVQuadTreeTmplBase *qt = this;
		while (qt->_datas.is_null()) {
			qt = qt->get_node(e);
		}

		// insertion
		qt->_datas.push_back(e);

		// does the current node must be splitted?
		if((qt->_datas.size() >= MAX_SIZE) && (qt->_cur_level < qt->_max_level)) {
			qt->create_next_level();
		}
	}

	void insert_list(const entry &e)
	{
		_datas.push_back(e);
	}
public:
	virtual void do_sub_insert(entry &/*e*/) = 0;

	virtual PVQuadTreeTmplBase *get_node(const entry &e) = 0;

	virtual void create_next_level() = 0;

	virtual int compute_index(const entry &e) = 0;

	Data  _datas;
	unsigned _max_level;
	unsigned _cur_level;
};

template <class Data, int DEPTH>
class PVQuadTreeTmpl : public PVQuadTreeTmplBase<Data>
{
public:
	PVQuadTreeTmpl() {}

	PVQuadTreeTmpl(uint32_t y1_min_value, uint32_t y1_max_value, uint32_t y2_min_value, uint32_t y2_max_value, int max_level, int cur_level = 0) :
		_y1_min_value(y1_min_value),
		_y1_max_value(y1_max_value),
		_y2_min_value(y2_min_value),
		_y2_max_value(y2_max_value)
	{
		PVQuadTreeTmplBase<Data>::set(max_level, cur_level);
		_y1_mid_value = (_y1_min_value + _y1_max_value) / 2;
		_y2_mid_value = (_y2_min_value + _y2_max_value) / 2;
	}

	~PVQuadTreeTmpl()
	{
	}

	void set(uint32_t y1_min_value, uint32_t y1_max_value, uint32_t y2_min_value, uint32_t y2_max_value, int max_level, int cur_level = 0)
	{
		PVQuadTreeTmplBase<Data>::set(max_level, cur_level);
		_y1_min_value = y1_min_value;
		_y1_max_value = y1_max_value;
		_y2_min_value = y2_min_value;
		_y2_max_value = y2_max_value;
		_y1_mid_value = (_y1_min_value + _y1_max_value) / 2;
		_y2_mid_value = (_y2_min_value + _y2_max_value) / 2;
	}

	bool compare(PVQuadTree<Data> &qt)
	{
		if(this->_datas.is_null() && qt._datas.is_null()) {
			// *this sont des noeuds, on va voir les
			// noeuds fils
			return _nodes[0].compare(*qt._nodes[0])
				&& _nodes[1].compare(*qt._nodes[1])
				&& _nodes[2].compare(*qt._nodes[2])
				&& _nodes[3].compare(*qt._nodes[3]);
		} else if(this->_datas.is_null()) {
			// *this et qt ne sont pas de même type
			return false;
		} else if(this->_datas.size() != qt._datas.size()) {
			// *this et qt sont de même type mais
			// pas avec le même nombre d'éléments
			return false;
		} else {
			// même type et même nombre d'éléments
			for(unsigned i = 0; i < this->_datas.size(); ++i) {
				if(are_diff(this->_datas.at(i), qt._datas.at(i))) {
					return false;
				}
			}
			return true;
		}
	}

private:

	PVQuadTreeTmplBase<Data> *get_node(const entry &e)
	{
		return &(_nodes[compute_index(e)]);
	}

	void do_sub_insert(entry &e)
	{
		// std::cout << "PVQuadTreeTmpl::do_sub_insert()" << std::endl;
		_nodes[compute_index(e)].insert(e);
	}

	void create_next_level()
	{
		// std::cout << "PVQuadTreeTmpl::create_next_level()" << std::endl;
		_nodes[NE].set(_y1_mid_value, _y1_max_value,
		                _y2_mid_value, _y2_max_value,
		                this->_max_level, this->_cur_level + 1);

		_nodes[SE].set(_y1_mid_value, _y1_max_value,
		               _y2_min_value, _y2_mid_value,
		               this->_max_level, this->_cur_level + 1);

		_nodes[SW].set(_y1_min_value, _y1_mid_value,
		               _y2_min_value, _y2_mid_value,
		               this->_max_level, this->_cur_level + 1);

		_nodes[NW].set(_y1_min_value, _y1_mid_value,
		               _y2_mid_value, _y2_max_value,
		               this->_max_level, this->_cur_level + 1);

		for(unsigned i = 0; i < this->_datas.size(); ++i) {
			entry const& e = this->_datas.at(i);
			_nodes[compute_index(e)].insert_list(e);
			//_nodes[compute_index(e)].insert(e);
			//_nodes[compute_index(e)]._datas.push_back(e);
		}
		this->_datas.clear();
	}

	inline int compute_index(const entry &e)
	{
		return ((e.y2 > _y2_mid_value) << 1) | (e.y1 > _y1_mid_value);
	}

public:
	PVQuadTreeTmpl<Data, DEPTH-1> _nodes[4];

	uint32_t     _y1_min_value;
	uint32_t     _y1_max_value;
	uint32_t     _y2_min_value;
	uint32_t     _y2_max_value;

	uint32_t     _y1_mid_value;
	uint32_t     _y2_mid_value;
};


template <class Data>
class PVQuadTreeTmpl<Data, 0> : public PVQuadTreeTmplBase<Data>
{
public:
	PVQuadTreeTmpl()
	{
	}

	//PVQuadTreeTmpl(uint32_t y1_min_value, uint32_t y1_max_value, uint32_t y2_min_value, uint32_t y2_max_value, int max_level, int cur_level = 0) :

	PVQuadTreeTmpl(uint32_t, uint32_t, uint32_t, uint32_t, int, int)
	{
	}

	~PVQuadTreeTmpl()
	{
	}

	void set(uint32_t, uint32_t, uint32_t, uint32_t, int max_level, int cur_level)
	{
		// std::cout << "PVQuadTreeTmpl<0>::set()" << std::endl;
		PVQuadTreeTmplBase<Data>::set(max_level, cur_level);
	}

	bool compare(PVQuadTree<Data> &qt)
	{
		if(this->_datas.is_null() && qt._datas.is_null()) {
			// *this sont des noeuds, on va voir les
			// noeuds fils mais ce n'est pas possible
			// car niveau "0"
			return false;
		} else if(this->_datas.is_null()) {
			// *this et qt ne sont pas de même type
			return false;
		} else if(this->_datas.size() != qt._datas.size()) {
			// *this et qt sont de même type mais
			// pas avec le même nombre d'éléments
			return false;
		} else {
			// même type et même nombre d'éléments
			for(unsigned i = 0; i < this->_datas.size(); ++i) {
				if(are_diff(this->_datas.at(i), qt._datas.at(i))) {
					return false;
				}
			}
			return true;
		}
	}

	void do_sub_insert(entry &/*e*/) { }

	PVQuadTreeTmplBase<Data> *get_node(const entry &/*e*/) { return 0; }

	void create_next_level() {}
	int compute_index(const entry &/*e*/) { return 0; }

public:
};

#endif // QUADTREE_TMPL_H
