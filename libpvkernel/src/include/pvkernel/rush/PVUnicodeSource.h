/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVRUSH_PVUNICODESOURCE_FILE_H
#define PVRUSH_PVUNICODESOURCE_FILE_H

#include <pvkernel/rush/PVCharsetDetect.h>
#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/rush/PVInput.h>

extern "C" {
#include <unicode/ucsdet.h>
#include <unicode/ucnv.h>
}

namespace PVRush {

template < template <class T> class Allocator = PVCore::PVMMapAllocator >

	/**
	 * Source to read file in unicode format.
	 *
	 * It creates Chunks/Elements/Fields without BOM and in UTF-8
	 */
class PVUnicodeSource : public PVRawSourceBase {
public:
	using alloc_chunk = Allocator<char>;
	using PVChunkAlloc = PVCore::PVChunkMem<Allocator>;
	using map_offsets = std::map<chunk_index, input_offset>;

public:
	PVUnicodeSource(PVInput_p input, size_t chunk_size, const alloc_chunk &alloc = alloc_chunk()) :
		PVRawSourceBase(), _chunk_size(chunk_size), _input(input),
		_curc(nullptr), _nextc(nullptr), _alloc(alloc)
	{
		assert(chunk_size > 10);
		assert(input);
		_offsets[0] = 0;
		seek_begin();
	}

	/**
	 * Disable copy/move as we don't want to handle chunk duplication.
	 */
	PVUnicodeSource(PVUnicodeSource const&) = delete;
	PVUnicodeSource(PVUnicodeSource &&) = delete;
	PVUnicodeSource& operator=(PVUnicodeSource const&) = delete;
	PVUnicodeSource& operator=(PVUnicodeSource &&) = delete;

	/**
	 * Clean up in progress chunks.
	 */
	~PVUnicodeSource(){
		if (_curc) {
			_curc->free();
		}
		if (_nextc && _nextc != _curc) {
			_nextc->free();
		}
	}

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
	PVCore::PVChunk* operator()() override
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
		_nextc = PVChunkAlloc::allocate(_chunk_size, this, _alloc);
		if (_nextc == NULL) {
			PVLOG_ERROR("(PVRawSource) unable to get a new chunk: end of input\n");
			return NULL;
		}

		return ret;
	}

	void seek_begin() override
	{
		_input->seek_begin();
		if (_curc)
			_curc->free();
		if (_nextc && _nextc != _curc)
			_nextc->free();
		_curc = PVChunkAlloc::allocate(_chunk_size, this, _alloc);
		_nextc = PVChunkAlloc::allocate(_chunk_size, this, _alloc);
	}

	bool seek(input_offset off) override
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

	QString human_name() override
	{
		return _input->human_name();
	}

	void release_input() override
	{
		_input->release();
	}

	func_type f() { return boost::bind<PVCore::PVChunk*>(&PVUnicodeSource::operator(), this); }

	void prepare_for_nelts(chunk_index /*nelts*/) override { }

	input_offset get_input_offset_from_index(chunk_index idx, chunk_index& known_idx) override
	{
		auto it = std::find_if(_offsets.begin(), _offsets.end(), [idx](std::pair<chunk_index, input_offset> const& a)
																		   { return a.first < idx; });
		if(it == _offsets.end()) {
			it = --_offsets.end();
		}
		known_idx = it->first;
		return it->second; // We don't know that index yet, start from the last known input offset
	}

private:
	size_t _chunk_size; //!< Size of the chunk
	PVInput_p _input; //!< Input source where we read data
	map_offsets _offsets; //!< Map indexes to input offsets
	PVCore::PVChunk* _curc; //!< Pointer to current chunk.
	PVCore::PVChunk* _nextc; //!< Pointer to next chunk.
	PVCharsetDetect _cd; //!< Charset detector
	alloc_chunk _alloc; //!< Allocator to create chunks
};

}

#endif
