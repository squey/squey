/**
 * \file PVSparseSelection.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PICVIZ_PVSPARSESELECTION_H
#define PICVIZ_PVSPARSESELECTION_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/PVBitCount.h>

#include "boost/integer/static_log2.hpp"

#include <tbb/tbb_allocator.h>

#include <map>

namespace Picviz {

class LibPicvizDecl PVSparseSelection
{
public:
	typedef uint32_t chunk_index_t;
	typedef uint64_t chunk_t;

	typedef std::map<chunk_index_t, chunk_t, std::less<chunk_index_t>, tbb::tbb_allocator<std::pair<chunk_index_t, chunk_t> > > map_chunks_t;
	//typedef std::map<chunk_index_t, chunk_t> map_chunks_t;

private:
	constexpr static size_t nbits_per_chunk = sizeof(chunk_t)*8;
	constexpr static size_t nbits_per_chunk_ln2 = boost::static_log2<nbits_per_chunk>::value;

public:
	PVSparseSelection()
	{
	}

	~PVSparseSelection() { }

public:
	inline void clear() { _chunks.clear(); }

	inline void set(size_t bit)
	{
		chunk_t& c = get_chunk_by_idx(bit_to_chunk_index(bit));
		c |= 1UL<<(bit_to_chunk_bit(bit));
	}

	void unset(size_t bit)
	{
		const chunk_index_t c = bit_to_chunk_index(bit);
		if (chunk_exists(c)) {
			get_last_chunk() &= ~(1UL<<(bit_to_chunk_bit(bit)));
		}
	}

	size_t bit_count() const
	{
		size_t ret = 0;
		map_chunks_t::const_iterator it;
		for (it = _chunks.begin(); it != _chunks.end(); it++) {
			ret += PVCore::PVBitCount::bit_count(it->second);
		}
		return ret;
	}

	bool is_set(size_t bit) const
	{
		const chunk_index_t c = bit_to_chunk_index(bit);
		map_chunks_t::const_iterator it_c;
		if (!chunk_exists(c, it_c)) {
			return false;
		}
		return (it_c->second) & (1UL<<(bit_to_chunk_bit(bit)));
	}

	template <class F>
	void visit_selected_lines(F const& f) const
	{
		map_chunks_t::const_iterator it;
		for (it = _chunks.begin(); it != _chunks.end(); it++) {
			const chunk_t c = it->second;
			if (c == 0) {
				continue;
			}
			const size_t off = ((size_t)(it->first))<<nbits_per_chunk_ln2;
			size_t nbits = PVCore::PVBitCount::bit_count(c);
			for (size_t i = 0; ((i < 64) & (nbits > 0)); i++) {
				if ((c & (1UL<<(i)))) {
					f(off + i);
					nbits--;
				}
			}
			/*const size_t off = ((size_t)(it->first))<<nbits_per_chunk_ln2;
			for (size_t i = 0; i < nbits_per_chunk; i++) {
				if ((c & (1UL<<(i)))) {
					f(off + i);
				}
			}*/
		}
	}

	template <class L>
	inline void to_list(L& l) const
	{
		visit_selected_lines([=,&l](size_t b) { l.push_back(b); } );
	}

	inline map_chunks_t const& get_chunks() const { return _chunks; }

public:
	PVSparseSelection& operator&=(PVSparseSelection const& o);
	PVSparseSelection& operator|=(PVSparseSelection const& o);
	//PVSpareSelection& operator^=(PVSpareSelection const& o);
	//

private:
	// Internal helper functions
	
	// Chunk manipulations
	inline chunk_t& get_chunk_by_idx(const chunk_index_t idx)
	{
		if (_chunks.size() > 0 && get_last_chunk_index() == idx) {
			return get_last_chunk();
		}

		// Try to find this chunk.
		map_chunks_t::iterator it_c = _chunks.find(idx);
		if (it_c == _chunks.end()) {
			// This chunk does not exist. Add a new one. Specify the last used chunk as a helper in order
			// to potentially improve the insertion performance.
			if (_chunks.size() > 0) {
				it_c = insert_new_chunk_with_last(idx);
			}
			else {
				it_c = insert_new_chunk(idx);
			}
		}

		_last_chunk = it_c;
		return it_c->second;
	}

	inline bool chunk_exists(const chunk_index_t idx)
	{
		if (_chunks.size() > 0 && get_last_chunk_index() == idx) {
			return true;
		}

		map_chunks_t::iterator it_c = _chunks.find(idx);
		if (it_c == _chunks.end()) {
			return false;
		}
		_last_chunk = it_c;
		return true;
	}

	inline bool chunk_exists(const chunk_index_t idx, map_chunks_t::const_iterator& it_c) const
	{
		if (_chunks.size() > 0 && get_last_chunk_index() == idx) {
			it_c = _last_chunk;
			return true;
		}

		it_c = _chunks.find(idx);
		return it_c != _chunks.end();
	}

	inline chunk_index_t get_last_chunk_index() const { return _last_chunk->first; }
	inline chunk_t& get_last_chunk() { return _last_chunk->second; }
	inline chunk_t const& get_last_chunk() const { return _last_chunk->second; }

	inline map_chunks_t::iterator insert_new_chunk(const chunk_index_t idx)
	{ return std::move(_chunks.insert(std::make_pair(idx, (chunk_t)0)).first); }

	inline map_chunks_t::iterator insert_new_chunk_with_last(const chunk_index_t idx)
	{ return std::move(_chunks.insert(_last_chunk, std::make_pair(idx, (chunk_t)0))); }

	// Common bit computations
	inline static chunk_index_t bit_to_chunk_index(size_t bit) { return bit >> nbits_per_chunk_ln2; }
	inline static chunk_index_t bit_to_chunk_bit(size_t bit) { return bit & (nbits_per_chunk-1); }

private:
	map_chunks_t _chunks;
	map_chunks_t::iterator _last_chunk;
};

}

#endif
