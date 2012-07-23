/**
 * \file PVSparseSelection.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <picviz/PVSparseSelection.h>


Picviz::PVSparseSelection& Picviz::PVSparseSelection::operator&=(PVSparseSelection const& o)
{
	// Check for self-assignement
	if (&o == this) {
		return *this;
	}

	if (o.get_chunks().size() == 0) {
		clear();
		return *this;
	}

	map_chunks_t::iterator it_this; map_chunks_t::const_iterator it_o;
	it_this = _chunks.begin(); it_o = o._chunks.begin();

	const map_chunks_t::const_iterator it_o_end = o._chunks.end();
	
	chunk_index_t idx_this;
	while (it_this != _chunks.end()) {
		idx_this = it_this->first;
		chunk_index_t idx_o = it_o->first;
		// TODO: we can have a logarithmic number of camparaisons here !
		while (idx_o < idx_this && it_o != o._chunks.end()) {
			it_o++;
			idx_o = it_o->first;
		}
		if (idx_o == idx_this) {
			// We found the current index of our object in the other one. Let's do the merge.
			it_this->second &= it_o->second;
			it_this++; it_o++;
		}
		else {
			// This chunk is not in the other sel. So, remove it.
			map_chunks_t::iterator it_next = it_this; it_next++;
			_chunks.erase(it_this);
			it_this = it_next;
		}
	}

	return *this;
}

Picviz::PVSparseSelection& Picviz::PVSparseSelection::operator|=(PVSparseSelection const& o)
{
	// Check for self-assignement
	if (&o == this || o.get_chunks().size() == 0) {
		return *this;
	}

	map_chunks_t::iterator it_this, it_before_this; map_chunks_t::const_iterator it_o;
	it_this = _chunks.begin(); it_o = o._chunks.begin();

	it_before_this = _chunks.begin(); 
	std::advance(it_before_this, -1);
	
	chunk_index_t idx_this;
	while (it_this != _chunks.end()) {
		idx_this = it_this->first;
		chunk_index_t idx_o = it_o->first;
		while (idx_o < idx_this) {
			// idx_o does not exists in our object. Let's copy it.
			// The element position that precedes this new element in our object is it_before_this.
			_chunks.insert(it_before_this, *it_o);
			it_o++;
			idx_o = it_o->first;
		}
		if (idx_o == idx_this) {
			// We found the current index of our object in the other one. Let's do the merge.
			it_this->second |= it_o->second;
			it_o++;
		}
		it_before_this = it_this;
		it_this++;
	}
	for (; it_o != o._chunks.end(); it_o++) {
		_chunks.insert(it_before_this, *it_o);
	}

	return *this;
}
