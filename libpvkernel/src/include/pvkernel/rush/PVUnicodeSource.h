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
#include <pvkernel/rush/PVConverter.h>

namespace PVRush
{

/**
 * Exception throw from error with Unicode Source.
 */
class UnicodeSourceError : public std::runtime_error
{
	using std::runtime_error::runtime_error;
};

/**
 * Source to read file in unicode format.
 *
 * It creates Chunks/Elements/Fields without BOM and in UTF-8
 */
template <template <class T> class Allocator = PVCore::PVMMapAllocator>
class PVUnicodeSource : public PVRawSourceBase
{
  public:
	using alloc_chunk = Allocator<char>;
	using PVChunkAlloc = PVCore::PVChunkMem<Allocator>;
	using map_offsets = std::map<chunk_index, input_offset>;

  public:
	PVUnicodeSource(PVInput_p input, size_t chunk_size, const alloc_chunk& alloc = alloc_chunk())
	    : PVRawSourceBase()
	    , _chunk_size(chunk_size)
	    , _input(input)
	    , _alloc(alloc)
	    , _utf8_converter("UTF-8")
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
	PVUnicodeSource(PVUnicodeSource&&) = delete;
	PVUnicodeSource& operator=(PVUnicodeSource const&) = delete;
	PVUnicodeSource& operator=(PVUnicodeSource&&) = delete;

	/**
	 * Clean up in progress chunks.
	 */
	~PVUnicodeSource()
	{
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
		if (*(it - 1) == 0xd) {
			_curc->add_element(begin, it - 1)->set_physical_end(it);
		} else {
			_curc->add_element(begin, it)->set_physical_end(it);
		}
	}

	/**
	 * Split a full chunk in elements on new-line.
	 *
	 * @warning: we guess that end is an end of element.
	 */
	void create_elements(char* begin, char* end)
	{
		auto it = begin;
		while ((it = std::find(begin, end, '\n')) != end) {
			add_element(begin, it);
			begin = it + 1;
		}
		if (begin != end) {
			// Add a final elements if the last "part" is the end of file.
			add_element(begin, end);
		}
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
		unsigned char* bom24 = (unsigned char*)begin_read;
		if (bom24[0] == 0xEF && bom24[1] == 0xBB && bom24[2] == 0xBF) {
			return begin_read + 3;
		}
		uint16_t& bom16 = *((uint16_t*)begin_read);
		if (bom16 == 0xFFFE || bom16 == 0xFEFF) {
			return begin_read + 2;
		}
		return begin_read;
	}

	/**
	 * Detect charset and remove BOM if required.
	 */
	char* get_begin_from_charset(char* begin, size_t len)
	{
		if (!_cd.found() and _charset == "") {
			if (_cd.HandleData(begin, len) == NS_OK) {
				_cd.DataEnd();
				if (_cd.found()) {
					_charset = _cd.GetCharset();
					PVLOG_DEBUG("Encoding found : %s\n", _charset.c_str());

					bool remove_bom =
					    (_charset.find("UTF") != _charset.npos); // Remove BOM only for UTF-X
					if (remove_bom) {
						return begining_with_bom(begin);
					}

				} else {
					// If charset is not set, it is pure ascii. Included in UTF-8.
					_charset = "UTF-8";
					return begining_with_bom(begin);
				}
			}
		}
		return begin;
	}

	/**
	 * Detect end and return nullptr if not new line can be found.
	 */
	char* get_end_from_charset(char* begin, size_t len)
	{
		assert(not _charset.empty());

		int extra_char = 0;
		if (_charset.find("UTF-16") != _charset.npos) {
			extra_char = 1;
		} else if (_charset.find("UTF-32") != _charset.npos) {
			extra_char = 3;
		}

		// FIXME : This is a manual "reverse find" as reverse_iterator appear only with C++14
		char* last = nullptr;
		for (char* it = begin + len; it != begin; --it) {
			if (*it == '\n') {
				last = it;
				break;
			}
		}
		if (last == nullptr) {
			return nullptr;
		}

		assert(_chunk_size % 8 == 0 && "To be sure last + extra_char is in 'read' data");
		return last + 1 + extra_char;
	}

	/**
	 * Refill Chunk buffer with UTF-8 data and return new end position for the buffer
	 */
	char* convert_buffer(char* begin, char* end)
	{
		assert(not _charset.empty());
		if (_charset != "UTF-8") {
			_tmp_buf.resize(std::distance(begin, end));
			std::copy(begin, end, _tmp_buf.begin());

			if (not _origin_converter) {
				_origin_converter.reset(new PVConverter(_charset));
			}

			char* target = begin;
			const char* dest = &_tmp_buf.front();

			UErrorCode status = U_ZERO_ERROR;
			ucnv_convertEx(&_utf8_converter.get(), &_origin_converter->get(), &target,
			               _curc->physical_end(), &dest, dest + _tmp_buf.size(), NULL, NULL, NULL,
			               NULL, true, true, &status);
			if (U_FAILURE(status)) {
				throw UnicodeSourceError("Fail conversion from ICU");
			}
			return target;
		}

		return end;
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
				create_elements(_curc->begin(), _curc->end());

				_curc->set_elements_index();
				_curc->init_elements_fields();

				PVCore::PVChunk* ret = _curc;
				// _nextc is empty, so the next call to this function will return NULL
				_curc = _nextc;
				return ret;
			} else {
				// No more data to read
				return NULL;
			}
		}

		char* b = get_begin_from_charset(_curc->begin(), _curc->size() + r);
		char* end = get_end_from_charset(_curc->begin(), _curc->size() + r);
		char* buffer_end = _curc->end() + r;
		// Grow chunk until we have at least a new line in the chunk or end of file is reach.
		while (end == nullptr) {
			if (buffer_end != _curc->physical_end()) {
				// Case where there is only the last line in the chunk.
				end = buffer_end;
				break;
			}
			size_t start_offset = b - _curc->begin();
			_curc = _curc->realloc_grow(_chunk_size / 10);

			b = start_offset + _curc->begin();
			;
			r = _input->operator()(_curc->end(), _curc->avail());
			buffer_end = _curc->end() + r;

			end = get_end_from_charset(_curc->end(), r);
		}

		// Process the read data
		// Copy remaining chars in the next chunk
		_nextc->set_end(std::copy(end, buffer_end, _nextc->begin()));

		_curc->set_end(convert_buffer(b, end));

		// Create an element and align its end on Chunk's end
		create_elements(b, _curc->end());

		// Set the index of the elements inside the chunk
		_curc->set_elements_index();

		// Allocate memory for the fields
		_curc->init_elements_fields();

		// Compute the chunk indexes, based on the number of elements found
		chunk_index next_index = _curc->index() + _curc->c_elements().size();
		_offsets[next_index] = _input->current_input_offset();
		_nextc->set_index(next_index);
		if (next_index - 1 > _last_elt_index) {
			_last_elt_index = next_index - 1;
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

	QString human_name() override { return _input->human_name(); }

	void release_input() override { _input->release(); }

	func_type f() { return boost::bind<PVCore::PVChunk*>(&PVUnicodeSource::operator(), this); }

	void prepare_for_nelts(chunk_index /*nelts*/) override {}

	input_offset get_input_offset_from_index(chunk_index idx, chunk_index& known_idx) override
	{
		auto it = std::find_if(
		    _offsets.begin(), _offsets.end(),
		    [idx](std::pair<chunk_index, input_offset> const& a) { return a.first <= idx; });
		if (it == _offsets.end()) {
			it = --_offsets.end();
		}
		known_idx = it->first;
		return it->second; // We don't know that index yet, start from the last known input offset
	}

  private:
	size_t _chunk_size;                //!< Size of the chunk
	PVInput_p _input;                  //!< Input source where we read data
	map_offsets _offsets;              //!< Map indexes to input offsets
	PVCore::PVChunk* _curc = nullptr;  //!< Pointer to current chunk.
	PVCore::PVChunk* _nextc = nullptr; //!< Pointer to next chunk.
	alloc_chunk _alloc;                //!< Allocator to create chunks

	// Attribute for charset conversion to UTF-8
	PVCharsetDetect _cd;                            //!< Charset detector
	std::string _charset;                           //!< Detected charset
	PVConverter _utf8_converter;                    //!< ICU converter to UTF 8
	std::unique_ptr<PVConverter> _origin_converter; //!< ICU converter from origin charset
	std::string _tmp_buf;                           //!< Temporary buffer use for charset conversion
};
}

#endif
