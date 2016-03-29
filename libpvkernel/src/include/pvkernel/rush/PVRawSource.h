/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVRAWSOURCE_FILE_H
#define PVRAWSOURCE_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/rush/PVChunkAlign.h>
#include <pvkernel/rush/PVChunkTransform.h>
#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/rush/PVInput.h>
#include <memory>
#include <iostream>

#include <tbb/tbb_allocator.h>

namespace PVRush {

//template < template <class T> class Allocator = PVCore::PVMMapAllocator >
template < template <class T> class Allocator = tbb::scalable_allocator >
class PVRawSource : public PVRush::PVRawSourceBase {
public:
	typedef PVCore::PVChunkMem<Allocator> PVChunkAlloc;
	typedef Allocator<char> alloc_chunk;
	typedef PVRawSource<Allocator> PVRawSource_t;
	typedef std::map<chunk_index,input_offset> map_offsets;

public:
	PVRawSource(PVInput_p input, size_t chunk_size, const alloc_chunk &alloc = alloc_chunk()) :
		PVRawSourceBase(), _chunk_size(chunk_size), _input(input), _alloc(alloc)
	{
		assert(chunk_size > 10);
		assert(input);
		_offsets[0] = 0;
		_curc = NULL;
		_nextc = NULL;
		seek_begin();
	}

	virtual ~PVRawSource()
	{
		if (_curc)
			_curc->free();
		if (_nextc && _nextc != _curc)
			_nextc->free();
	}

public:
	void release_input() override
	{
		_input->release();
	}

public:
	PVCore::PVChunk* operator()()
	{
		// The current chunk must be filled with data, then aligned and returned
		
		// Read data from input
		char* begin_read = _curc->end();
		size_t r = _input->operator()(begin_read, _curc->avail());

		if (r == 0) { // No more data to read.
			if (_curc->size() > 0) {
				auto begin = _curc->begin();
				auto it = _curc->begin();
				while((it = std::find(begin, _curc->end(), '\n')) != _curc->end()) {
					if(*(it - 1) == 0xd) {
						_curc->add_element(begin, it - 1)->set_physical_end(_curc->physical_end());
					} else {
						_curc->add_element(begin, it)->set_physical_end(_curc->physical_end());
					}
					begin = it + 1;
				}
				if(begin != _curc->end()) {
					if(*(_curc->end() - 1) == 0xd) {
						_curc->add_element(begin, _curc->end() - 1)->set_physical_end(_curc->physical_end());
					} else {
						_curc->add_element(begin, _curc->end())->set_physical_end(_curc->physical_end());
					}
				}

				_curc->set_elements_index();
				_curc->init_elements_fields();

				PVCore::PVChunk* ret = _curc;
				// _nextc is empty, so the next call to this function will return NULL
				_curc = _nextc;
				return ret;
			}
			else {
				// No more data to read
				return NULL;
			}
		}
		size_t len_read = r + _curc->size();
		auto b = _curc->begin();
		// Check first four bytes
		// //FIXME : TODO : FIXME : TODO : FIXME : TODO : FIXME
		// It have to be done only for utfX, use chardetect to do it correctly!!
		if (len_read >= 4) {
			uint32_t bom = *((uint32_t*)begin_read);
			if (bom == 0xFFFE0000 || bom == 0x0000FEFF) {
				b += 4;
				len_read -= 4;
			}
		}
		if (len_read >= 3) {
			unsigned char* data_u = (unsigned char*) begin_read;
			if (data_u[0] == 0xEF && data_u[1] == 0xBB && data_u[2] == 0xBF) {
				b += 3;
				len_read -= 3;
			}
		}
		if (len_read >= 2) {
			uint16_t& bom = *((uint16_t*) begin_read);
			if (bom == 0xFFFE || bom == 0xFEFF) {
				b += 2;
				len_read -= 2;
			}
		}

		// Process the read data
		size_t size_read_processed = len_read;
		auto end = _curc->end()+r;

		// Create an element and align its end on Chunk's end
		auto begin = b;
		auto it = b;
		while((it = std::find(begin, end, '\n')) != end) {
			if(*(it - 1) == 0xd) {
				_curc->add_element(begin, it - 1)->set_physical_end(it);
			} else {
				_curc->add_element(begin, it)->set_physical_end(it);
			}
			begin = it + 1;
		}

		// Copy remaining chars in the next chunk
		_nextc->set_end(std::copy(begin, end, _nextc->begin()));

		_curc->set_end(begin);

		// Set the index of the elements inside the chunk
		_curc->set_elements_index();

		// Allocate memory for the fields
		_curc->init_elements_fields();

		// Compute the chunk indexes, based on the number of elements found
		chunk_index next_index = _curc->index() + _curc->c_elements().size();
		_offsets[next_index] =  _input->current_input_offset();
		_nextc->set_index(next_index);
		if (next_index-1>_last_elt_index) {
			_last_elt_index = next_index-1;
		}

		// Invert the chunks, allocate the new one and go on
		PVCore::PVChunk* ret = _curc;
		_curc = _nextc;
		_nextc = PVChunkAlloc::allocate(_chunk_size, (PVRawSourceBase*)this, _alloc);
		if (_nextc == NULL) {
			PVLOG_ERROR("(PVRawSource) unable to get a new chunk: end of input\n");
			return NULL;
		}

		return ret;
	}
public:
	static size_t get_ring_buffer_size(int nchunks, int chunk_size)
	{
		return (nchunks+2)*(chunk_size*sizeof(char)+sizeof(PVChunkAlloc));
	}

	virtual bool discover() { return false; }

	virtual void seek_begin()
	{
		_input->seek_begin();
		if (_curc)
			_curc->free();
		if (_nextc && _nextc != _curc)
			_nextc->free();
		_curc = PVChunkAlloc::allocate(_chunk_size, this, _alloc);
		_nextc = PVChunkAlloc::allocate(_chunk_size, this, _alloc);
	}

	virtual bool seek(input_offset off)
	{
		if (!_input->seek(off)) {
			return false;
		}
		if (_curc)
			_curc->free();
		if (_nextc && _nextc != _curc)
			_nextc->free();
		_curc = PVChunkAlloc::allocate(_chunk_size, this, _alloc);
		_nextc = PVChunkAlloc::allocate(_chunk_size, this, _alloc);
		return true;
	}

	virtual QString human_name()
	{
		return _input->human_name();
	}
	
	input_offset get_input_offset_from_index(chunk_index idx, chunk_index& known_idx)
	{
		map_offsets::iterator it;
		for (it = _offsets.begin(); it != _offsets.end(); it++) {
			chunk_index src_index = it->first;
			if (idx >= src_index) {
				known_idx = src_index;
				return it->second;
			}
		}
		it = _offsets.end(); it--;
		known_idx = it->first;
		return it->second; // We don't know that index yet, start from the last known input offset
	}

	virtual void prepare_for_nelts(chunk_index /*nelts*/) { }

	PVInput_p get_input() { return _input; }

	virtual func_type f() { return boost::bind<PVCore::PVChunk*>(&PVRawSource<Allocator>::operator(), this); }

protected:
	size_t _chunk_size; //!< Size of the chunk
	PVInput_p _input; //!< Input source where we read data
protected:
	mutable PVCore::PVChunk* _curc; //!< Pointer to current chunk.
	mutable PVCore::PVChunk* _nextc; //!< Pointer to next chunk.
	alloc_chunk _alloc;
	mutable map_offsets _offsets; // Map indexes to input offsets
};

}

#endif
