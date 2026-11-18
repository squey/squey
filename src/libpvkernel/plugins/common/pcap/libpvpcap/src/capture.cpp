//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <libpvpcap/capture.h>

#include <pvkernel/core/PVStreamingCompressor.h>

#include <QFileInfo>
#include <QString>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstring>
#include <tuple>

namespace pvpcap
{

namespace
{

constexpr uint32_t PCAP_MICROSECONDS = 0xa1b2c3d4;
constexpr uint32_t PCAP_NANOSECONDS = 0xa1b23c4d;
constexpr uint32_t PCAPNG_SECTION_HEADER = 0x0a0d0d0a;
constexpr uint32_t PCAPNG_BYTE_ORDER = 0x1a2b3c4d;

constexpr size_t PCAP_HEADER_SIZE = 24;
constexpr size_t PCAP_RECORD_HEADER_SIZE = 16;

constexpr uint32_t BLOCK_INTERFACE = 0x00000001;
constexpr uint32_t BLOCK_OBSOLETE_PACKET = 0x00000002;
constexpr uint32_t BLOCK_SIMPLE_PACKET = 0x00000003;
constexpr uint32_t BLOCK_ENHANCED_PACKET = 0x00000006;
constexpr uint32_t BLOCK_SYSTEMD_JOURNAL = 0x00000009;
constexpr uint32_t BLOCK_CUSTOM = 0x00000bad;
constexpr uint32_t BLOCK_CUSTOM_NO_COPY = 0x40000bad;

constexpr size_t READ_SIZE = 8 * 1024 * 1024;

/* No record of a real capture comes close: past this, the length was read from a file
 * that is damaged there. Wireshark stops at a few megabytes for most link types. */
constexpr uint32_t LARGEST_RECORD = 256 * 1024 * 1024;

uint32_t load32(const uint8_t* p)
{
	uint32_t v;
	std::memcpy(&v, p, sizeof(v));
	return v;
}

} // namespace

std::FILE* open_file(const std::string& path, const char* mode)
{
#ifdef _WIN32
	const std::wstring wide_path = QString::fromStdString(path).toStdWString();
	const std::wstring wide_mode = QString::fromLatin1(mode).toStdWString();
	return _wfopen(wide_path.c_str(), wide_mode.c_str());
#else
	return std::fopen(path.c_str(), mode);
#endif
}

capture_reader::capture_reader(const std::string& path)
{
	const QFileInfo info(QString::fromStdString(path));
	if (not info.isReadable()) {
		throw capture_error("Can't read '" + path + "'");
	}
	_size = info.size();
	const std::string extension = info.suffix().toLower().toStdString();
	for (const std::string& compression : PVCore::PVStreamingDecompressor::supported_extensions()) {
		_compressed |= extension == compression;
	}
	try {
		_input = std::make_unique<PVCore::PVStreamingDecompressor>(path);
	} catch (const std::exception& e) {
		throw capture_error("Can't read '" + path + "': " + e.what());
	}

	_buffer.resize(READ_SIZE);
	if (not fill(4)) {
		throw capture_error("'" + path + "' is empty");
	}

	const uint32_t magic = load32(_buffer.data());
	if (magic == PCAP_MICROSECONDS or magic == PCAP_NANOSECONDS) {
		_format = format::pcap;
		_nanoseconds = magic == PCAP_NANOSECONDS;
	} else if (std::byteswap(magic) == PCAP_MICROSECONDS or
	           std::byteswap(magic) == PCAP_NANOSECONDS) {
		_format = format::pcap;
		_nanoseconds = std::byteswap(magic) == PCAP_NANOSECONDS;
		_swapped = true;
	} else if (magic == PCAPNG_SECTION_HEADER) {
		_format = format::pcapng;
		return;
	} else {
		throw capture_error("'" + path + "' holds neither a pcap nor a pcapng capture");
	}

	if (not fill(PCAP_HEADER_SIZE)) {
		throw capture_error("'" + path + "' is cut short in its file header");
	}
	_header.assign(_buffer.begin(), _buffer.begin() + PCAP_HEADER_SIZE);
	// The upper bits of the field tell about the frame check sequences, not the link type.
	_link_type = u32(_header.data() + 20) & 0x03ffffff;
	_begin = PCAP_HEADER_SIZE;
	_position = PCAP_HEADER_SIZE;
}

capture_reader::~capture_reader() = default;

uint16_t capture_reader::u16(const uint8_t* p) const
{
	uint16_t v;
	std::memcpy(&v, p, sizeof(v));
	return _swapped ? std::byteswap(v) : v;
}

uint32_t capture_reader::u32(const uint8_t* p) const
{
	const uint32_t v = load32(p);
	return _swapped ? std::byteswap(v) : v;
}

uint64_t capture_reader::u64(const uint8_t* p) const
{
	uint64_t v;
	std::memcpy(&v, p, sizeof(v));
	return _swapped ? std::byteswap(v) : v;
}

capture_reader::pcapng_interface capture_reader::read_interface(const uint8_t* body,
                                                                size_t length) const
{
	pcapng_interface i;
	i.link_type = u16(body);
	i.snap_length = u32(body + 4);
	// Options: a code, a length, and the value padded to 32 bits, up to the end of options.
	for (size_t at = 8; at + 4 <= length;) {
		const uint16_t code = u16(body + at);
		const uint16_t size = u16(body + at + 2);
		if (code == 0 or at + 4 + size > length) {
			break;
		}
		const uint8_t* value = body + at + 4;
		if (code == 9 and size >= 1) { // if_tsresol
			i.binary = value[0] & 0x80;
			i.exponent = value[0] & 0x7f;
		} else if (code == 14 and size >= 8) { // if_tsoffset
			i.offset = static_cast<int64_t>(u64(value));
		}
		at += 4 + ((size + 3) & ~size_t(3));
	}
	return i;
}

int64_t capture_reader::nanoseconds(const pcapng_interface& i, uint64_t timestamp)
{
	__int128 ns;
	if (i.binary) {
		ns = (static_cast<__int128>(timestamp) * 1000000000) >> std::min<uint8_t>(i.exponent, 63);
	} else if (i.exponent <= 9) {
		__int128 scale = 1;
		for (int e = i.exponent; e < 9; e++) {
			scale *= 10;
		}
		ns = static_cast<__int128>(timestamp) * scale;
	} else {
		__int128 scale = 1;
		for (int e = 9; e < std::min<int>(i.exponent, 38); e++) {
			scale *= 10;
		}
		ns = static_cast<__int128>(timestamp) / scale;
	}
	return static_cast<int64_t>(ns + static_cast<__int128>(i.offset) * 1000000000);
}

/**
 * Makes `needed` bytes from _begin available in the buffer.
 *
 * @return false if the file ends before them.
 */
bool capture_reader::fill(size_t needed)
{
	if (_end - _begin >= needed) {
		return true;
	}
	if (_begin > 0) {
		std::memmove(_buffer.data(), _buffer.data() + _begin, _end - _begin);
		_end -= _begin;
		_begin = 0;
	}
	if (_buffer.size() < needed) {
		_buffer.resize(needed);
	}
	while (_end < needed and not _eof) {
		size_t read = 0;
		size_t compressed = 0;
		try {
			std::tie(read, compressed) =
			    _input->read(reinterpret_cast<char*>(_buffer.data() + _end), _buffer.size() - _end);
		} catch (const std::exception& e) {
			_damage = std::string("the capture can't be read further: ") + e.what();
			_eof = true;
			break;
		}
		_end += read;
		_file_position += compressed;
		if (read == 0) {
			_eof = true;
		}
	}
	return _end - _begin >= needed;
}

bool capture_reader::next(capture_record& record)
{
	if (not _damage.empty()) {
		return false;
	}
	return _format == format::pcap ? next_pcap(record) : next_pcapng(record);
}

bool capture_reader::next_pcap(capture_record& record)
{
	if (not fill(PCAP_RECORD_HEADER_SIZE)) {
		if (_end != _begin) {
			_damage = "the capture is cut short in the middle of a packet";
		}
		return false;
	}
	const uint8_t* header = _buffer.data() + _begin;
	const uint32_t captured = u32(header + 8);
	const uint32_t wire = u32(header + 12);
	if (captured > LARGEST_RECORD) {
		_damage = "the capture is damaged: a packet claims " + std::to_string(captured) + " bytes";
		return false;
	}
	const size_t size = PCAP_RECORD_HEADER_SIZE + captured;
	if (not fill(size)) {
		_damage = "the capture is cut short in the middle of a packet";
		return false;
	}

	const uint8_t* bytes = _buffer.data() + _begin;
	record.type = capture_record::kind::packet;
	record.bytes = {bytes, size};
	record.frame = {bytes + PCAP_RECORD_HEADER_SIZE, captured};
	record.link_type = _link_type;
	record.wire_length = wire;
	record.timestamp =
	    int64_t(u32(bytes)) * 1000000000 + int64_t(u32(bytes + 4)) * (_nanoseconds ? 1 : 1000);

	_begin += size;
	_position += size;
	return true;
}

bool capture_reader::next_pcapng(capture_record& record)
{
	if (not fill(12)) {
		if (_end != _begin) {
			_damage = "the capture is cut short in the middle of a block";
		}
		return false;
	}

	const uint8_t* bytes = _buffer.data() + _begin;
	const uint32_t type = load32(bytes);
	if (type == PCAPNG_SECTION_HEADER) {
		// Each section states its own byte order, after the length it applies to.
		const uint32_t byte_order = load32(bytes + 8);
		if (byte_order == PCAPNG_BYTE_ORDER) {
			_swapped = false;
		} else if (std::byteswap(byte_order) == PCAPNG_BYTE_ORDER) {
			_swapped = true;
		} else {
			_damage = "the capture is damaged: a section header has no byte order";
			return false;
		}
		_interfaces.clear();
	}

	const uint32_t length = u32(bytes + 4);
	if (length < 12 or length % 4 != 0 or length > LARGEST_RECORD) {
		_damage = "the capture is damaged: a block claims " + std::to_string(length) + " bytes";
		return false;
	}
	if (not fill(length)) {
		_damage = "the capture is cut short in the middle of a block";
		return false;
	}
	bytes = _buffer.data() + _begin;
	if (u32(bytes + length - 4) != length) {
		_damage = "the capture is damaged: a block does not end where it says";
		return false;
	}

	const uint8_t* body = bytes + 8;
	const size_t body_length = length - 12;
	const uint32_t block_type = u32(bytes);

	record = capture_record{};
	record.bytes = {bytes, length};

	const auto packet = [&](uint32_t interface, size_t data_offset, uint32_t captured,
	                        uint32_t wire, uint64_t timestamp) {
		if (interface >= _interfaces.size()) {
			_damage = "the capture is damaged: a packet names an interface it does not describe";
			return false;
		}
		if (data_offset > body_length or captured > body_length - data_offset) {
			_damage = "the capture is damaged: a packet claims more bytes than its block holds";
			return false;
		}
		record.type = capture_record::kind::packet;
		record.frame = {body + data_offset, captured};
		record.link_type = _interfaces[interface].link_type;
		record.wire_length = wire;
		record.timestamp = nanoseconds(_interfaces[interface], timestamp);
		return true;
	};
	const auto timestamp_at = [&](const uint8_t* p) {
		return (uint64_t(u32(p)) << 32) | u32(p + 4);
	};

	bool valid = true;
	switch (block_type) {
	case BLOCK_INTERFACE:
		if (body_length < 8) {
			_damage = "the capture is damaged: an interface block is too short";
			return false;
		}
		_interfaces.push_back(read_interface(body, body_length));
		break;
	case BLOCK_ENHANCED_PACKET:
		valid = body_length >= 20 and
		        packet(u32(body), 20, u32(body + 12), u32(body + 16), timestamp_at(body + 4));
		break;
	case BLOCK_OBSOLETE_PACKET:
		valid = body_length >= 20 and
		        packet(u16(body), 20, u32(body + 12), u32(body + 16), timestamp_at(body + 4));
		break;
	case BLOCK_SIMPLE_PACKET: {
		if (body_length < 4 or _interfaces.empty()) {
			valid = false;
			break;
		}
		const uint32_t wire = u32(body);
		// A simple packet block holds as much of the packet as the snapshot length kept,
		// padded to 32 bits: its length is its only bound when there is no snapshot length.
		uint32_t captured = std::min<uint32_t>(wire, body_length - 4);
		if (_interfaces[0].snap_length != 0) {
			captured = std::min(captured, _interfaces[0].snap_length);
		}
		// A simple packet block has no timestamp, which tshark shows as the epoch.
		valid = packet(0, 4, captured, wire, 0);
		record.timestamp = 0;
		break;
	}
	case BLOCK_SYSTEMD_JOURNAL:
	case BLOCK_CUSTOM:
	case BLOCK_CUSTOM_NO_COPY:
		record.type = capture_record::kind::event;
		break;
	default:
		// Blocks of system call captures (sysdig, falco) number 0x200 and up.
		if ((block_type & 0xffffff00) == 0x00000200) {
			_splittable = false;
		}
		record.type = capture_record::kind::metadata;
		break;
	}
	if (not valid) {
		if (_damage.empty()) {
			_damage = "the capture is damaged: a packet block is too short";
		}
		return false;
	}

	_begin += length;
	_position += length;
	return true;
}

} // namespace pvpcap
