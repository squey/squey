/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVRUSH_PVUNICODESOURCE_FILE_H
#define PVRUSH_PVUNICODESOURCE_FILE_H

#include <pvkernel/rush/PVCharsetDetect.h>
#include <pvkernel/rush/PVConverter.h>
#include <pvkernel/rush/PVInput.h>
#include <pvkernel/rush/PVInputFile.h>
#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/rush/PVUnicodeSourceError.h>

namespace PVRush
{

/**
 * Exception thrown by Unicode Source when the conversion result
 * would not fit in the supplied buffer.
 */
class UnicodeSourceBufferOverflowError : public std::exception
{
	using std::exception::exception;
};

/**
 * Source to read file in unicode format.
 *
 * It creates Chunks/Elements/Fields without BOM and in UTF-8
 */
template <template <class T> class Allocator = PVCore::PVMMapAllocator>
class PVUnicodeSource : public PVRawSourceBaseType<PVCore::PVTextChunk>
{
  public:
	using alloc_chunk = Allocator<char>;
	using PVChunkAlloc = PVCore::PVTextChunkMem<Allocator>;

  public:
	static constexpr const char MULTI_INPUTS_SEPARATOR = ';';

  public:
	PVUnicodeSource(PVInput_p input, size_t chunk_size, const alloc_chunk& alloc = alloc_chunk())
	    : _chunk_size(chunk_size), _input(input), _alloc(alloc), _utf8_converter("UTF-8")
	{
		assert(chunk_size > 10);
		assert(input);
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
	~PVUnicodeSource() override
	{
		if (_curc) {
			_curc->free();
		}
		if (_nextc && _nextc != _curc) {
			_nextc->free();
		}
	}

	/**
	 * Size in Mo of the file handled by this source.
	 */
	size_t get_size() const override
	{
		return dynamic_cast<PVRush::PVInputFile*>(_input.get())->file_size();
	}

	/**
	 * Add element [begin, it[ as current chunk elements.
	 *
	 * @note Remove \r for \r\n new-line (windows style)
	 */
	virtual void add_element(char* begin, char* it)
	{
		if (*(it - 1) == 0xd) {
			_curc->add_element(begin, it - 1)->set_physical_end(it);
		} else {
			_curc->add_element(begin, it)->set_physical_end(it);
		}
	}

	/**
	 * Add an uninitialized element with an allocated buffer
	 */
	virtual PVCore::PVElement* add_uninitialized_element(size_t n)
	{
		PVCore::PVElement* elt = _curc->add_element();
		elt->allocate_new(n);
		return elt;
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
	 * Return the endianness suffix of UTF charsets in order to avoid the
	 * need to add a BOM in the begining of each chunk.
	 * (uchardet doesn't specify it anymore since version 0.0.5 because
	 * the standard explicitly warns against it)
	 */
	static const char* get_UTF_endianness_suffix_from_BOM(char* begin_read,
	                                                      const std::string& charset)
	{
		if (charset == "UTF-32") {
			uint32_t bom32 = *((uint32_t*)begin_read);
			if (bom32 == 0xFFFE0000) {
				return "BE";
			} else if (bom32 == 0x0000FEFF) {
				return "LE";
			}
		} else if (charset == "UTF-16") {
			uint16_t& bom16 = *((uint16_t*)begin_read);
			if (bom16 == 0xFFFE) {
				return "BE";
			} else if (bom16 == 0xFEFF) {
				return "LE";
			}
		}

		return "";
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

					bool remove_bom = false;
					if (_charset.find("UTF") != _charset.npos) {
						remove_bom = true;
						_charset += get_UTF_endianness_suffix_from_BOM(begin, _charset);
					}

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
	 * Detect end and return nullptr if no new line can be found.
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
			char* target_limit = _curc->physical_end();

			const char* source = &_tmp_buf.front();
			const char* source_limit = source + _tmp_buf.size();

			UErrorCode status = U_ZERO_ERROR;

			PVConverter target_converter(
			    ucnv_clone(&_utf8_converter.get(), &status));
			PVConverter source_converter(
			    ucnv_clone(&_origin_converter->get(), &status));

			status = U_ZERO_ERROR;

			// http://icu-project.org/apiref/icu4c/ucnv_8h.html#af4c967c5afa207d064c24e19256586b6
			ucnv_convertEx(&target_converter.get(), &source_converter.get(), &target, target_limit,
			               &source, source_limit,
			               nullptr, // pivotStart
			               nullptr, // pivotSource
			               nullptr, // pivotTarget
			               nullptr, // pivotLimit
			               true,    // reset
			               true,    // flush
			               &status);

			if (status == U_BUFFER_OVERFLOW_ERROR) {
				// Copy the unconverted string back to the chunk to prevent side effects
				std::copy(_tmp_buf.begin(), _tmp_buf.end(), begin);
				throw UnicodeSourceBufferOverflowError();
			} else if (U_FAILURE(status)) {
				throw UnicodeSourceError(
				    (std::string("ICU charset conversion failed : ") + u_errorName(status))
				        .c_str());
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
	PVCore::PVTextChunk* operator()() override
    {
        // Try to read data into current chunk
        char* begin_read = _curc->end();
        size_t r, c;
        std::tie(r, c) = _input->operator()(begin_read, _curc->avail());
        size_t chunk_size = r;
        size_t compressed_chunk_size = c;

        // End of input
        if (r == 0) {
            if (_curc->size() == 0) {
                return nullptr;
            }
            create_elements(_curc->begin(), _curc->end());
            _curc->init_elements_fields();
            PVCore::PVTextChunk* ret = _curc;
            _curc = _nextc;
            return ret;
        }

        char* b = get_begin_from_charset(_curc->begin(), _curc->size() + r);
        char* buffer_end = _curc->end() + r;
        char* end = get_end_from_charset(_curc->begin(), _curc->size() + r);

        // Ensure we find a line ending or EOF
        while (end == nullptr) {
            if (buffer_end == _curc->physical_end()) {
                size_t start_offset = b - _curc->begin();
                _curc = _curc->realloc_grow(_chunk_size / 10);
                b = _curc->begin() + start_offset;
                chunk_size = 0;
            } else if (chunk_size < _curc->avail()) {
                std::tie(r, c) = _input->operator()(_curc->end() + chunk_size, _curc->avail() - chunk_size);
                chunk_size += r;
                compressed_chunk_size += c;
                buffer_end = _curc->end() + chunk_size;
                if (r == 0) {
                    end = buffer_end; // EOF
                } else {
                    end = get_end_from_charset(buffer_end - r, r);
                }
            } else {
                end = buffer_end;
            }
        }

        // Copy overflow into next chunk
        if (end < buffer_end) {
            _nextc->set_end(std::copy(end, buffer_end, _nextc->begin()));
            double keep_ratio = 1.0 - double(buffer_end - end) / chunk_size;
            _curc->set_init_size(compressed_chunk_size * keep_ratio);
        }

        // Convert buffer, realloc if needed
        while (true) {
            try {
                _curc->set_end(convert_buffer(b, end));
                break;
            } catch (const UnicodeSourceBufferOverflowError&) {
                _curc = _curc->realloc_grow(_chunk_size);
                b = get_begin_from_charset(_curc->begin(), _curc->size() + r);
                end = get_end_from_charset(_curc->begin(), _curc->size() + r);
            }
        }

        // Build elements
        create_elements(b, _curc->end());
        _curc->init_elements_fields();

        // Update indexes
        chunk_index next_index = _curc->index() + _curc->c_elements().size();
        _nextc->set_index(next_index);
        _last_elt_index = std::max(_last_elt_index, next_index - 1);

        // Swap chunks
        PVCore::PVTextChunk* ret = _curc;
        _curc = _nextc;
        _nextc = PVChunkAlloc::allocate(_chunk_size, this, _alloc);
        if (!_nextc) {
            PVLOG_ERROR("(PVRawSource) unable to allocate new chunk\n");
            return nullptr;
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

	void release_input(bool cancel_first) override
	{
		if (cancel_first)
			_input->cancel();
	}

	void prepare_for_nelts(chunk_index /*nelts*/) override {}

  protected:
	size_t _chunk_size; //!< Size of the chunk
	PVInput_p _input;   //!< Input source where we read data

  private:
	PVCore::PVTextChunk* _curc = nullptr;  //!< Pointer to current chunk.
	PVCore::PVTextChunk* _nextc = nullptr; //!< Pointer to next chunk.
	alloc_chunk _alloc;                    //!< Allocator to create chunks

	// Attribute for charset conversion to UTF-8
	PVCharsetDetect _cd;                            //!< Charset detector
	std::string _charset;                           //!< Detected charset
	PVConverter _utf8_converter;                    //!< ICU converter to UTF 8
	std::unique_ptr<PVConverter> _origin_converter; //!< ICU converter from origin charset
	std::string _tmp_buf;                           //!< Temporary buffer use for charset conversion
};
} // namespace PVRush

#endif
