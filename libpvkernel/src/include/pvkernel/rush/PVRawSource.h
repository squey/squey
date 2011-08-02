#ifndef PVRAWSOURCE_FILE_H
#define PVRAWSOURCE_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/rush/PVChunkAlign.h>
#include <pvkernel/rush/PVChunkTransform.h>
#include <pvkernel/filter/PVRawSourceBase.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVFilterFunction.h>
#include <pvkernel/rush/PVInput.h>
#include <memory>

#include <tbb/scalable_allocator.h>

namespace PVRush {

template < template <class T> class Allocator = tbb::scalable_allocator >
class PVRawSource : public PVFilter::PVRawSourceBase {
public:
	typedef PVCore::PVChunkMem<Allocator> PVChunkAlloc;
	typedef Allocator<char> alloc_chunk;

	typedef std::map<PVCore::chunk_index,PVInput::input_offset> map_offsets;

public:
	PVRawSource(PVInput_p input,  PVChunkAlign &align, size_t chunk_size, PVChunkTransform &chunk_transform, PVFilter::PVChunkFilter_f src_filter, const alloc_chunk &alloc = alloc_chunk()) :
		PVRawSourceBase(src_filter), _input(input), _align(align), _chunk_size(chunk_size), _transform(chunk_transform), _alloc(alloc) 
	{
		assert(chunk_size > 10);
		assert(input);
		_curc = NULL;
		_nextc = NULL;
		seek_begin();
		// The original offset
		_offsets[0] = 0;
	}

	virtual ~PVRawSource()
	{
		if (_curc)
			_curc->free();
		if (_nextc && _nextc != _curc)
			_nextc->free();
	}
public:
	PVCore::PVChunk* operator()()
	{
		// The current chunk must be filled with data, then aligned and returned
		
		// Read data from input
		char* begin_read = _curc->end();
		size_t r = _input->operator()(begin_read, _transform.next_read_size(_curc->avail()));
		if (r == 0) {
			if (_curc->size() > 0) {
				// Create a final element with what's currently in the chunk and returns it
				_align_base(*_curc, *_nextc);
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
		// Process the read data
		size_t size_read_processed = _transform(begin_read, r, (size_t) ((uintptr_t)_curc->physical_end()-(uintptr_t)begin_read));

		_curc->set_end(_curc->end()+size_read_processed);

		// Align the chunk thanks to the next one if necessarry
		while (!_align(*_curc, *_nextc)) {
			// Ok, this chunk is too small, let's resize it !
			size_t grow_by = _chunk_size/10;
			_curc = _curc->realloc_grow(grow_by);
			char* previous_end = _curc->end();
			r = _input->operator()(previous_end, _transform.next_read_size(_curc->avail()));
			if (r == 0) {
				// Create a final element with what's in the chunk
				_align_base(*_curc, *_nextc);
				break;
			}
			size_read_processed = _transform(previous_end, r, (size_t) ((uintptr_t)_curc->physical_end()-(uintptr_t)previous_end));
			_curc->set_end(previous_end+size_read_processed);
		}

		// Set the index of the elements inside the chunk
		_curc->set_elements_index();

		// Compute the chunk indexes, based on the number of elements found
		PVCore::chunk_index next_index = _curc->index() + _curc->c_elements().size();
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

	PVRush::PVInput_p get_input() { return _input; }

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
	virtual QString human_name() { return _input->human_name(); }

protected:
	PVInput_p _input;
	PVChunkAlign &_align;
	PVChunkAlign _align_base;
	size_t _chunk_size;
	PVChunkTransform &_transform;
	mutable map_offsets _offsets;
protected:
	mutable PVCore::PVChunk* _curc;
	mutable PVCore::PVChunk* _nextc;
	alloc_chunk _alloc;
};

}

#endif
