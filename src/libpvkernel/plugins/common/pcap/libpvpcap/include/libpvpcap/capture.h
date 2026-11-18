/* * MIT License
 *
 * © Squey, 2026
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

#ifndef LIBPVPCAP_CAPTURE_H
#define LIBPVPCAP_CAPTURE_H

#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace PVCore
{
class PVStreamingDecompressor;
} // namespace PVCore

namespace pvpcap
{

struct capture_error : public std::runtime_error {
	using std::runtime_error::runtime_error;
};

/**
 * Opens a file whose path is in UTF-8, whatever the code page of the process.
 *
 * @return nullptr if it can't be opened, as std::fopen() does.
 */
std::FILE* open_file(const std::string& path, const char* mode);

/**
 * One record of a pcap or pcapng file, as the file stores it.
 *
 * The spans point into the buffer of the reader that returned the record, and stay valid
 * until its next call to capture_reader::next().
 */
struct capture_record {
	enum class kind {
		packet,   //!< a frame holding a captured packet
		event,    //!< a frame holding something else, such as a log entry or a custom block
		metadata, //!< no frame, but what the frames that follow need: interfaces, secrets...
	};

	kind type = kind::metadata;
	std::span<const uint8_t> bytes; //!< the whole record
	std::span<const uint8_t> frame; //!< the captured bytes of a packet
	uint32_t link_type = 0;         //!< of a packet
	uint32_t wire_length = 0;       //!< of a packet, before the capture cut it short
	int64_t timestamp = 0;          //!< of a packet, in nanoseconds since 1970, UTC
};

/**
 * Reads a pcap or pcapng file, one record at a time.
 *
 * Tshark numbers the frames of a capture in the order of their records: the records this
 * returns as packets and events are the frames tshark numbers, in the same order.
 *
 * A compressed file is read as the import of a text file reads it, decompressed on the way.
 */
class capture_reader
{
  public:
	enum class format { pcap, pcapng };

	/**
	 * @throw capture_error if the file can't be opened, or holds neither a pcap nor a pcapng
	 *        capture: tshark reads other formats, which can't be split.
	 */
	explicit capture_reader(const std::string& path);
	~capture_reader();

	capture_reader(const capture_reader&) = delete;
	capture_reader& operator=(const capture_reader&) = delete;

	format file_format() const { return _format; }

	/**
	 * What a capture holding some of the records of this one starts with: the file header
	 * of a pcap file. A pcapng file has none, its section headers come as metadata records.
	 */
	std::span<const uint8_t> header() const { return _header; }

	/**
	 * Reads the next record.
	 *
	 * @return false at the end of the file, or where it is damaged: damage() tells which.
	 */
	bool next(capture_record& record);

	//! What stopped the reading before the end of the file, empty if nothing did.
	const std::string& damage() const { return _damage; }

	//! Bytes of the capture read so far, once decompressed.
	uint64_t position() const { return _position; }

	//! Bytes of the file read so far, which are fewer than position() if it is compressed.
	uint64_t file_position() const { return _file_position; }

	//! Size of the file, in bytes.
	uint64_t size() const { return _size; }

	//! Whether the file is compressed, and the capture larger than it.
	bool compressed() const { return _compressed; }

	//! Of a pcap file: whether its byte order is not the one of this machine.
	bool swapped() const { return _swapped; }

	//! Of a pcap file: whether its timestamps count nanoseconds rather than microseconds.
	bool nanoseconds() const { return _nanoseconds; }

	//! Of a pcap file.
	uint32_t link_type() const { return _link_type; }

	/**
	 * Whether a part of the records makes a capture tshark reads as it does the whole.
	 *
	 * Not for a system call capture: its frames depend on process tables, which its metadata
	 * blocks hold as they go.
	 */
	bool splittable() const { return _splittable; }

  private:
	// Named so: Windows headers define "interface" as a macro.
	struct pcapng_interface {
		uint32_t link_type = 0;
		uint32_t snap_length = 0;
		bool binary = false; //!< resolution of the timestamps: 2^-exponent rather than 10^-
		uint8_t exponent = 6;
		int64_t offset = 0; //!< seconds added to the timestamps
	};

	bool fill(size_t needed);
	bool next_pcap(capture_record& record);
	bool next_pcapng(capture_record& record);
	uint16_t u16(const uint8_t* p) const;
	uint32_t u32(const uint8_t* p) const;
	uint64_t u64(const uint8_t* p) const;
	pcapng_interface read_interface(const uint8_t* body, size_t length) const;
	static int64_t nanoseconds(const pcapng_interface& i, uint64_t timestamp);

  private:
	std::unique_ptr<PVCore::PVStreamingDecompressor> _input;
	format _format = format::pcap;
	bool _swapped = false;
	std::vector<uint8_t> _header;
	std::vector<uint8_t> _buffer;
	size_t _begin = 0; //!< first byte of the buffer not returned yet
	size_t _end = 0;   //!< end of the bytes read into the buffer
	bool _eof = false;
	uint64_t _position = 0;
	uint64_t _file_position = 0;
	uint64_t _size = 0;
	uint32_t _link_type = 0;   //!< of a pcap file
	bool _nanoseconds = false; //!< of a pcap file

	std::vector<pcapng_interface> _interfaces; //!< of the pcapng section
	std::string _damage;
	bool _splittable = true;
	bool _compressed = false;
};

/**
 * Writes a capture of records read from others.
 *
 * Writes nothing of its own: a pcap file is to be given its header first, and a pcapng file
 * a section header before anything else.
 */
class capture_writer
{
  public:
	//! @throw capture_error if the file can't be created
	explicit capture_writer(const std::string& path);
	~capture_writer();

	capture_writer(const capture_writer&) = delete;
	capture_writer& operator=(const capture_writer&) = delete;

	//! @throw capture_error if it could not be written
	void write(std::span<const uint8_t> bytes);

	//! @throw capture_error if what was written could not all be
	void close();

  private:
	std::string _path;
	std::FILE* _file = nullptr;
	std::vector<uint8_t> _buffer;
};

/**
 * Blocks of a pcapng file, in the byte order of this machine.
 */
namespace pcapng
{
//! A section header, which the interfaces and packets of the section follow.
std::vector<uint8_t> section_header();

//! An interface whose timestamps count microseconds, or nanoseconds.
std::vector<uint8_t>
interface_description(uint32_t link_type, uint32_t snap_length, bool nanoseconds);

//! A packet of a pcap file, given as it is stored: record header in its byte order, then data.
std::vector<uint8_t>
packet_of_pcap(std::span<const uint8_t> record, bool swapped, bool nanoseconds);
} // namespace pcapng

} // namespace pvpcap

#endif // LIBPVPCAP_CAPTURE_H
