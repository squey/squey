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
#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/rush/PVCharsetDetect.h>
#include <pvkernel/rush/PVInput.h>
#include <memory>
#include <iostream>
#include <iterator>

#include <tbb/tbb_allocator.h>

extern "C" {
#include <unicode/ucsdet.h>
#include <unicode/ucnv.h>
}

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

	/**
	 * Add element [begin, it[ as current chunk elements.
	 *
	 * @note Remove \r for \r\n new-line (windows style)
	 */
	void add_element(char* begin, char* it)
	{
		if(*(it - 1) == 0xd) {
			_curc->add_element(begin, it - 1)->set_physical_end(it);
		} else {
			_curc->add_element(begin, it)->set_physical_end(it);
		}
	}

	/**
	 * Split a full chunk in elements on new-line.
	 *
	 * @return fist char which can't be put in a new element.
	 */
	char* create_elements(char* begin, char* end) {
		auto it = begin;
		while((it = std::find(begin, end, '\n')) != end) {
			add_element(begin, it);
			begin = it + 1;
		}
		return begin;
	}

	/**
	 * Return beginning of the buffer skipping bom values.
	 */
	static char* begining_with_bom(char* begin_read)
	{
		// Check first four bytes
		// //FIXME : TODO : FIXME : TODO : FIXME : TODO : FIXME
		// It have to be done only for utfX, use chardetect to do it correctly!!
		uint32_t bom32 = *((uint32_t*)begin_read);
		if (bom32 == 0xFFFE0000 || bom32 == 0x0000FEFF) {
			return begin_read + 4;
		}
		unsigned char* bom24 = (unsigned char*) begin_read;
		if (bom24[0] == 0xEF && bom24[1] == 0xBB && bom24[2] == 0xBF) {
			return begin_read + 3;
		}
		uint16_t& bom16 = *((uint16_t*) begin_read);
		if (bom16 == 0xFFFE || bom16 == 0xFEFF) {
			return begin_read + 2;
		}
		return begin_read;
	}

	/**
	 * Generate a new Chunk.
	 *
	 * Continue with previous created chunk.
	 * * Read data from source
	 * * Split it on new lines and create elements with it
	 * * Add remaining character in the next chunk.
	 * * Computed indexes.
	 */
	PVCore::PVChunk* operator()()
	{
		// The current chunk must be filled with data, then aligned and returned
		
		// Read data from input
		char* begin_read = _curc->end();
		size_t r = _input->operator()(begin_read, _curc->avail());

		if (r == 0) { // No more data to read.
			if (_curc->size() > 0) {
				char* begin = create_elements(_curc->begin(), _curc->end());
				if(begin != _curc->end()) { // Handle final line without new line.
					add_element(begin, _curc->end());
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

		char* b = _curc->begin();

		UErrorCode status = U_ZERO_ERROR;

		static std::string charset;
		if (!_cd.found() and charset == "") {
			if (_cd.HandleData(b, r) == NS_OK) {
				_cd.DataEnd();
				if (_cd.found()) {
					charset = _cd.GetCharset();
					PVLOG_DEBUG("Encoding found : %s\n", charset.c_str());
					bool remove_bom = false;
					if (charset.find("UTF") != charset.npos) // Remove BOM only for UTF-X
						remove_bom = true;

					if(remove_bom)
						b = begining_with_bom(b);
				} else {
					// If charset is not set, it is pure ascii. Included in UTF-8.
					charset = "UTF-8";
					b = begining_with_bom(b);
				}
			}
		}

		int extra_char = 0;
		if(charset.find("UTF-16") != charset.npos)
		{
			extra_char = 1;
		} else if(charset.find("UTF-32") != charset.npos) {
			extra_char = 3;
		}

		char* last = nullptr;
		for(char* it=_curc->end() + r; it!=_curc->begin(); --it) {
			if(*it == '\n') {
				last = it;
				break;
			}
		}
		if(last == nullptr)
		{
			// TODO : We should increase chunk size and do it all over again.
			throw std::runtime_error("No new line in the read chunk. You need bigger chunk size");
		}

		// Process the read data
		auto end = _curc->end()+r;

		// Copy remaining chars in the next chunk
		_nextc->set_end(std::copy(last + 1 + extra_char, end, _nextc->begin()));

		_curc->set_end(last + 1 + extra_char);

		if(charset != "UTF-8") {
			size_t tmp_size = _curc->size();
			std::unique_ptr<char[]> tmp_dest(new char[tmp_size]);
			std::copy(b, _curc->end(), tmp_dest.get());

			UConverter *utf8Cnv = ucnv_open("UTF-8", &status);

			if(U_FAILURE(status)) {
				throw std::runtime_error("Fail conversion 1");
			}

			UConverter* cnv = ucnv_open(charset.c_str(), &status);
			if(U_FAILURE(status)) {
				throw std::runtime_error("Fail conversion 2");
			}
			char* target = b;
			const char* dest = tmp_dest.get();

			ucnv_convertEx(utf8Cnv, cnv, &target, _curc->physical_end(), &dest, tmp_dest.get() + tmp_size, NULL, NULL, NULL, NULL, true, true, &status);
			if (U_FAILURE(status)) {
				throw std::runtime_error("Fail conversion");
			}
			_curc->set_end(target);
			ucnv_close(cnv);
			ucnv_close(utf8Cnv);
		}

		// Create an element and align its end on Chunk's end
		create_elements(b, _curc->end());

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
	PVCore::PVChunk* _curc; //!< Pointer to current chunk.
	PVCore::PVChunk* _nextc; //!< Pointer to next chunk.
	alloc_chunk _alloc;
	map_offsets _offsets; // Map indexes to input offsets
	mutable PVCharsetDetect _cd; //!< Charset detector : FIXME Should exist only for Text files !!!
};

}

#endif
