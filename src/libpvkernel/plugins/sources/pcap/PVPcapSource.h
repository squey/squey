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

#ifndef __PVPCAPSOURCE_FILE_H__
#define __PVPCAPSOURCE_FILE_H__

#include <pvkernel/core/PVLogger.h>
#include <pvkernel/core/PVAllocators.h>
#include <pvkernel/core/PVTextChunk.h>
#include <pvkernel/rush/PVInput.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/rush/PVRawSourceBase.h>

#include "../../common/pcap/PVPcapDescription.h"
#include "../../common/pcap/libpvpcap/include/libpvpcap.h"
#include "../../common/pcap/libpvpcap/include/libpvpcap/extraction.h"

#include <QFileInfo>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace PVRush
{

/**
 * The frames of a capture, one element each: the number of the frame, its conversation, the
 * name of the capture if several are imported together, then what tshark wrote of it.
 *
 * Reading the capture starts with the first chunk asked for, rather than when the source is
 * created: the extractor creates the sources of all its inputs at once, and reads them one
 * after the other.
 */
class PVPcapSource : public PVRawSourceBaseType<PVCore::PVTextChunk>
{
	using chunk_memory = PVCore::PVTextChunkMem<PVCore::PVMMapAllocator>;

  public:
	PVPcapSource(PVPcapDescription& input, size_t chunk_size)
	    : _input(input), _capture_size(QFileInfo(_input.path()).size()), _chunk_size(chunk_size)
	{
		if (_input.multi_inputs()) {
			_prefix = QFileInfo(_input.path()).fileName().toStdString() + pvpcap::SEPARATOR;
		}
	}

	~PVPcapSource() override = default;

  public:
	QString human_name() override { return _input.human_name(); }

	void seek_begin() override
	{
		set_extraction(nullptr);
		_pending = false;
		_ended = false;
		_next_index = 0;
	}

	void prepare_for_nelts(chunk_index /*nelts*/) override {}

	size_t get_size() const override { return _capture_size; }

	// What tshark got of the capture: the chunks only tell it as the frames a display filter
	// keeps fill them, and the lines of the frames come in the order of the capture, as the
	// slowest of the tshark processes the capture is split between write them.
	size_t read_size() const override
	{
		if (_ended) {
			return _capture_size;
		}
		const std::lock_guard<std::mutex> lock(_extraction_mutex);
		return _extraction ? std::min<size_t>(_extraction->read_bytes(), _capture_size) : 0;
	}

	void release_input(bool /*cancel_first*/) override { set_extraction(nullptr); }

	PVCore::PVTextChunk* operator()() override
	{
		if (_ended) {
			return nullptr;
		}
		try {
			if (not _extraction) {
				start();
			}
			return read_chunk();
		} catch (const pvpcap::extraction_error& e) {
			set_extraction(nullptr);
			throw PVInputException("tshark could not read '" + _input.path().toStdString() +
			                       "':\n\n" + e.what());
		}
	}

  private:
	void start()
	{
		if (not _input.can_extract()) {
			throw PVInputException(
			    "The capture '" + _input.path().toStdString() +
			    "' was imported by a version of Squey that did not record what it asked "
			    "tshark: import it again.");
		}
		const auto offsets = _input.offsets();
		if (not offsets) {
			throw PVInputException("The captures before '" + _input.path().toStdString() +
			                       "' were not read");
		}
		_frames_offset = offsets->first;
		_streams_offset = offsets->second;

		pvpcap::extraction::options opts = _input.extraction();
		opts.work_directory = PVNrawCacheManager::nraw_dir().toStdString();
		set_extraction(std::make_unique<pvpcap::extraction>(_input.path().toStdString(), opts));
	}

	PVCore::PVTextChunk* read_chunk()
	{
		PVCore::PVTextChunk* chunk = chunk_memory::allocate(_chunk_size, this);
		if (chunk == nullptr) {
			throw PVInputException("Can't allocate memory for the frames");
		}
		size_t bytes = 0;
		size_t rows = 0;
		while (true) {
			if (not _pending) {
				bool more = false;
				try {
					more = _extraction->next(_line);
				} catch (...) {
					chunk->free();
					throw;
				}
				if (not more) {
					break;
				}
				_pending = true;
			}
			char number[24];
			char stream[16];
			char* number_end =
			    std::to_chars(number, number + sizeof(number), _frames_offset + _line.frame).ptr;
			char* stream_end =
			    _line.stream == pvpcap::NO_STREAM
			        ? stream
			        : std::to_chars(stream, stream + sizeof(stream), _streams_offset + _line.stream)
			              .ptr;
			const size_t size = (number_end - number) + 1 + (stream_end - stream) + 1 +
			                    _prefix.size() + _line.fields.size();
			if (size > chunk->avail()) {
				if (rows > 0) {
					break;
				}
				chunk = chunk->realloc_grow(size);
			}

			char* begin = chunk->end();
			char* end = std::copy(number, number_end, begin);
			*end++ = pvpcap::SEPARATOR[0];
			end = std::copy(stream, stream_end, end);
			*end++ = pvpcap::SEPARATOR[0];
			end = std::copy(_prefix.begin(), _prefix.end(), end);
			end = std::copy(_line.fields.begin(), _line.fields.end(), end);
			chunk->add_element(begin, end);
			chunk->set_end(end);

			bytes += _line.bytes;
			rows++;
			_pending = false;
		}

		if (rows == 0) {
			chunk->free();
			finish();
			return nullptr;
		}
		chunk->set_index(_next_index);
		_next_index += rows;
		_last_elt_index = _next_index - 1;
		chunk->set_init_size(bytes);
		chunk->init_elements_fields();
		return chunk;
	}

	void finish()
	{
		_ended = true;
		_input.read(_extraction->frames(), _extraction->streams());
		const std::string damage = _extraction->damage();
		if (not damage.empty()) {
			PVLOG_WARN("'%s': %s\n", qPrintable(_input.path()), damage.c_str());
		}
		set_extraction(nullptr);
	}

	//! Ends the previous extraction out of the lock of read_size(): it waits for tshark.
	void set_extraction(std::unique_ptr<pvpcap::extraction> extraction)
	{
		std::unique_ptr<pvpcap::extraction> previous;
		const std::lock_guard<std::mutex> lock(_extraction_mutex);
		previous = std::exchange(_extraction, std::move(extraction));
	}

  private:
	PVPcapDescription& _input;
	const size_t _capture_size;
	size_t _chunk_size;
	std::string _prefix; //!< name of the capture, when several are imported together
	//! Only the thread of the source changes it, under the mutex, for read_size().
	std::unique_ptr<pvpcap::extraction> _extraction;
	mutable std::mutex _extraction_mutex;
	pvpcap::extraction::line _line;
	bool _pending = false; //!< whether _line did not fit in the previous chunk
	std::atomic<bool> _ended = false;
	uint64_t _frames_offset = 0;
	uint64_t _streams_offset = 0;
	chunk_index _next_index = 0;
};

} // namespace PVRush

#endif // __PVPCAPSOURCE_FILE_H__
