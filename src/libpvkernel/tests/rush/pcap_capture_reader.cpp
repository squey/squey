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

// Captures are read record by record, as tshark numbers their frames.
//
// The import splits a capture by copying its records as they are, and numbers the frames as
// tshark does: a pcapng block it took for a frame and tshark did not, or the other way
// round, would shift every frame number after it. A capture cut short gives its whole
// records and says where it stops.

#include <pvkernel/core/squey_assert.h>

#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap/capture.h"
#include "pcap_captures.h"
#include "common.h"

#include <QTemporaryDir>

#include <bit>
#include <iostream>

using namespace pcap_captures;
using kind = pvpcap::capture_record::kind;

namespace
{

//! A pcapng block, in the byte order of this machine.
bytes block(uint32_t type, const bytes& body)
{
	bytes b;
	const auto put = [&](uint32_t v) {
		const auto* p = reinterpret_cast<const uint8_t*>(&v);
		b.insert(b.end(), p, p + 4);
	};
	bytes padded = body;
	while (padded.size() % 4 != 0) {
		padded.push_back(0);
	}
	const uint32_t length = static_cast<uint32_t>(12 + padded.size());
	put(type);
	put(length);
	b.insert(b.end(), padded.begin(), padded.end());
	put(length);
	return b;
}

template <typename T>
void put_native(bytes& b, T v)
{
	const auto* p = reinterpret_cast<const uint8_t*>(&v);
	b.insert(b.end(), p, p + sizeof(v));
}

bytes section_header()
{
	bytes body;
	put_native<uint32_t>(body, 0x1a2b3c4d);
	put_native<uint16_t>(body, 1);
	put_native<uint16_t>(body, 0);
	put_native<int64_t>(body, -1);
	return block(0x0a0d0d0a, body);
}

//! An interface whose timestamps count nanoseconds.
bytes interface_description(uint16_t link_type)
{
	bytes body;
	put_native<uint16_t>(body, link_type);
	put_native<uint16_t>(body, 0);
	put_native<uint32_t>(body, 262144);
	put_native<uint16_t>(body, 9); // if_tsresol
	put_native<uint16_t>(body, 1);
	put_native<uint32_t>(body, 9);
	put_native<uint32_t>(body, 0);
	return block(1, body);
}

bytes enhanced_packet(uint32_t interface, uint64_t nanoseconds, const bytes& data)
{
	bytes body;
	put_native<uint32_t>(body, interface);
	put_native<uint32_t>(body, static_cast<uint32_t>(nanoseconds >> 32));
	put_native<uint32_t>(body, static_cast<uint32_t>(nanoseconds));
	put_native<uint32_t>(body, static_cast<uint32_t>(data.size()));
	put_native<uint32_t>(body, static_cast<uint32_t>(data.size() + 100));
	body.insert(body.end(), data.begin(), data.end());
	return block(6, body);
}

void write(const std::string& path, const bytes& content)
{
	std::ofstream(path, std::ios::binary)
	    .write(reinterpret_cast<const char*>(content.data()), std::streamsize(content.size()));
}

std::vector<pvpcap::capture_record> read_all(pvpcap::capture_reader& reader,
                                             std::vector<bytes>& frames)
{
	std::vector<pvpcap::capture_record> records;
	pvpcap::capture_record record;
	while (reader.next(record)) {
		records.push_back(record);
		frames.emplace_back(record.frame.begin(), record.frame.end());
	}
	return records;
}

} // namespace

int main()
{
	pvtest::init_ctxt();
	QTemporaryDir dir;
	PV_ASSERT_VALID(dir.isValid(), "a temporary directory", "could not be made");

	const bytes packet = ethernet(0x0800, ipv4({10, 0, 0, 1}, {10, 0, 0, 2}, UDP, udp(1, 2, {})));

	std::cout << "pcap" << std::endl;
	{
		const std::string path = dir.filePath("a.pcap").toStdString();
		write_pcap(path, 1, {{100, 5, packet}, {101, 999999, packet}});
		pvpcap::capture_reader reader(path);
		std::vector<bytes> frames;
		const auto records = read_all(reader, frames);
		PV_VALID(records.size(), size_t(2));
		PV_VALID(reader.damage(), std::string());
		PV_VALID(records[1].type == kind::packet, true);
		PV_VALID(records[1].link_type, uint32_t(1));
		PV_VALID(records[1].timestamp, int64_t(101999999000));
		PV_VALID(frames[0] == packet, true);
		PV_VALID(reader.header().size(), size_t(24));
	}

	std::cout << "pcap cut short" << std::endl;
	{
		const std::string whole = dir.filePath("whole.pcap").toStdString();
		write_pcap(whole, 1, {{100, 0, packet}, {101, 0, packet}});
		std::ifstream in(whole, std::ios::binary);
		bytes content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
		content.resize(content.size() - 10);
		const std::string cut = dir.filePath("cut.pcap").toStdString();
		write(cut, content);
		pvpcap::capture_reader reader(cut);
		std::vector<bytes> frames;
		PV_VALID(read_all(reader, frames).size(), size_t(1));
		PV_ASSERT_VALID(not reader.damage().empty(), "the end of a capture cut short",
		                "went unnoticed");
	}

	std::cout << "pcapng" << std::endl;
	{
		const bytes cooked =
		    linux_cooked(0x0800, ipv4({10, 0, 0, 1}, {10, 0, 0, 2}, UDP, udp(1, 2, {})));
		const std::string path = dir.filePath("a.pcapng").toStdString();
		write(path, section_header() + interface_description(113) +
		                enhanced_packet(0, 1700000000123456789ULL, cooked) +
		                block(0x00000bad, bytes(8, 1)) + // custom block, a frame for tshark
		                block(5, bytes(16, 0)) +         // interface statistics, no frame
		                interface_description(1) + enhanced_packet(1, 42, packet));
		pvpcap::capture_reader reader(path);
		PV_VALID(reader.file_format() == pvpcap::capture_reader::format::pcapng, true);
		std::vector<bytes> frames;
		const auto records = read_all(reader, frames);
		PV_VALID(reader.damage(), std::string());
		PV_VALID(records.size(), size_t(7));
		PV_VALID(records[0].type == kind::metadata and records[1].type == kind::metadata, true);
		PV_VALID(records[2].type == kind::packet, true);
		PV_VALID(records[2].link_type, uint32_t(113));
		PV_VALID(records[2].timestamp, int64_t(1700000000123456789LL));
		PV_VALID(records[2].wire_length, uint32_t(cooked.size() + 100));
		PV_VALID(frames[2] == cooked, true);
		PV_VALID(records[3].type == kind::event, true);
		PV_VALID(records[4].type == kind::metadata, true);
		PV_VALID(records[6].link_type, uint32_t(1));
		PV_VALID(frames[6] == packet, true);
		PV_VALID(reader.splittable(), true);
	}

	std::cout << "pcapng damaged" << std::endl;
	{
		bytes content = section_header() + interface_description(1) + enhanced_packet(0, 1, packet);
		// The trailing length of the last block no longer matches its leading one.
		content[content.size() - 4] ^= 0x10;
		const std::string path = dir.filePath("damaged.pcapng").toStdString();
		write(path, content);
		pvpcap::capture_reader reader(path);
		std::vector<bytes> frames;
		PV_VALID(read_all(reader, frames).size(), size_t(2));
		PV_ASSERT_VALID(not reader.damage().empty(), "a damaged block", "went unnoticed");
	}

	std::cout << "system calls" << std::endl;
	{
		const std::string path = dir.filePath("scap.pcapng").toStdString();
		write(path, section_header() + block(0x201, bytes(8, 0)));
		pvpcap::capture_reader reader(path);
		std::vector<bytes> frames;
		read_all(reader, frames);
		PV_VALID(reader.splittable(), false);
	}

	std::cout << "not a capture" << std::endl;
	{
		const std::string path = dir.filePath("text.pcap").toStdString();
		write(path, bytes(100, 'a'));
		bool refused = false;
		try {
			pvpcap::capture_reader reader(path);
		} catch (const pvpcap::capture_error&) {
			refused = true;
		}
		PV_VALID(refused, true);
	}

	return 0;
}
