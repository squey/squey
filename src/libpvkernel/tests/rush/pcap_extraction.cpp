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

// Split between several tshark, a capture reads as it does with a single one.
//
// The frames of a conversation go to the same tshark, but a tshark numbers the frames it
// reads, and times the first of them: the numbers of the frames fields name, the time since
// the first frame and the Info column are given back as a single tshark would have written
// them. This compares every line, whether the parts are fed through pipes or written to
// files first, for a capture of HTTP and DNS over IPv4, IPv6 and VLAN, on Ethernet and on
// a Linux "any" link, in a time zone other than UTC, with and without a display filter.
//
// Read twice, as tshark does for the fields that name later frames, a capture reads as it
// does with a single tshark reading it twice, the responses and the reassemblies a single
// pass can't know of included.

#include <pvkernel/core/PVUtils.h>
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNrawCacheManager.h>

#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap.h"
#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap/extraction.h"
#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap/ws.h"
#include "pcap_captures.h"
#include "common.h"

#include <QDir>
#include <QFileInfo>
#include <QTemporaryDir>

#include <algorithm>
#include <iostream>
#include <map>
#include <tuple>

using namespace pcap_captures;
using kind = pvpcap::extraction::field::kind;

namespace
{

constexpr uint32_t ETHERNET = 1;
constexpr uint32_t LINUX_SLL = 113;
constexpr uint32_t START = 1700000000; // 2023-11-14T22:13:20Z

bytes dns_name(const std::string& name)
{
	bytes b;
	size_t begin = 0;
	while (begin < name.size()) {
		const size_t end = std::min(name.find('.', begin), name.size());
		b.push_back(static_cast<uint8_t>(end - begin));
		b.insert(b.end(), name.begin() + begin, name.begin() + end);
		begin = end + 1;
	}
	b.push_back(0);
	return b;
}

bytes dns(uint16_t id, bool response, const std::string& name)
{
	bytes b;
	put16(b, id);
	put16(b, response ? 0x8180 : 0x0100);
	put16(b, 1);
	put16(b, response ? 1 : 0);
	put16(b, 0);
	put16(b, 0);
	b = b + dns_name(name);
	put16(b, 1);
	put16(b, 1);
	if (response) {
		put16(b, 0xc00c);
		put16(b, 1);
		put16(b, 1);
		put32(b, 300);
		put16(b, 4);
		b = b + bytes{192, 0, 2, static_cast<uint8_t>(id)};
	}
	return b;
}

struct packet {
	uint64_t microseconds; //!< since START
	int conversation;
	int order;
	bytes network; //!< from the IP header
	uint16_t type;
	std::vector<uint16_t> vlans;
};

std::vector<packet> conversations()
{
	std::vector<packet> packets;
	for (int c = 0; c < 48; c++) {
		const bool v6 = c % 3 == 1;
		const std::vector<uint16_t> vlans =
		    c % 3 == 2 ? std::vector<uint16_t>{10} : std::vector<uint16_t>{};
		const uint16_t port = static_cast<uint16_t>(40000 + c);
		const auto ip = [&](bool from_client, const bytes& segment) {
			if (v6) {
				const ipv6_address client = ipv6(0x2001, 0x0db8, static_cast<uint16_t>(c + 1));
				const ipv6_address server = ipv6(0x2a00, 0x1450, static_cast<uint16_t>(c % 5 + 1));
				return from_client ? pcap_captures::ipv6(client, server, TCP, segment)
				                   : pcap_captures::ipv6(server, client, TCP, segment);
			}
			const ipv4_address client{10, 0, 0, static_cast<uint8_t>(c + 1)};
			const ipv4_address server{192, 168, 1, static_cast<uint8_t>(c % 5 + 1)};
			return from_client ? ipv4(client, server, TCP, segment)
			                   : ipv4(server, client, TCP, segment);
		};
		const std::string request =
		    "GET /object/" + std::to_string(c) + " HTTP/1.1\r\nHost: www.example\r\n\r\n";
		const std::string body(1800 + 37 * c, 'x');
		const std::string response =
		    "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
		uint32_t client_seq = 1000;
		uint32_t server_seq = 5000;
		int order = 0;
		uint64_t t = uint64_t(c) * 700;
		const auto add = [&](bool from_client, uint8_t flags, const std::string& data) {
			const bytes segment = from_client ? tcp(port, 80, client_seq, server_seq, flags, data)
			                                  : tcp(80, port, server_seq, client_seq, flags, data);
			packets.push_back({t, c, order++, ip(from_client, segment),
			                   static_cast<uint16_t>(v6 ? 0x86dd : 0x0800), vlans});
			(from_client ? client_seq : server_seq) +=
			    static_cast<uint32_t>(data.size()) + ((flags & (SYN | FIN)) ? 1 : 0);
			t += 130;
		};
		add(true, SYN, "");
		add(false, SYN | ACK, "");
		add(true, ACK, "");
		add(true, PSH | ACK, request);
		add(false, ACK, response.substr(0, 1448));
		add(false, PSH | ACK, response.substr(1448));
		add(true, ACK, "");
		add(true, FIN | ACK, "");
		add(false, FIN | ACK, "");
		add(true, ACK, "");
	}
	for (int d = 0; d < 24; d++) {
		const ipv4_address client{10, 0, 1, static_cast<uint8_t>(d + 1)};
		const ipv4_address resolver{10, 0, 0, 53};
		const std::string name = "host" + std::to_string(d) + ".example";
		const uint16_t port = static_cast<uint16_t>(20000 + d);
		const uint64_t t = uint64_t(d) * 1500 + 50;
		packets.push_back(
		    {t,
		     100 + d,
		     0,
		     ipv4(client, resolver, UDP, udp(port, 53, dns(uint16_t(d), false, name))),
		     0x0800,
		     {}});
		packets.push_back({t + 400,
		                   100 + d,
		                   1,
		                   ipv4(resolver, client, UDP, udp(53, port, dns(uint16_t(d), true, name))),
		                   0x0800,
		                   {}});
	}
	std::sort(packets.begin(), packets.end(), [](const packet& a, const packet& b) {
		return std::tie(a.microseconds, a.conversation, a.order) <
		       std::tie(b.microseconds, b.conversation, b.order);
	});
	return packets;
}

void write_capture(const std::string& path, uint32_t link)
{
	std::vector<frame> frames;
	for (const packet& p : conversations()) {
		const bytes data = link == ETHERNET ? ethernet(p.type, p.network, p.vlans)
		                                    : linux_cooked(p.type, p.network);
		frames.push_back(
		    {START + uint32_t(p.microseconds / 1000000), uint32_t(p.microseconds % 1000000), data});
	}
	write_pcap(path, link, frames);
}

pvpcap::extraction::options profile_options(const std::string& work)
{
	pvpcap::extraction::options opts;
	opts.fields = {{"frame.number", kind::text},
	               {"frame.len", kind::text},
	               {"frame.time", kind::absolute_time},
	               {"frame.time_relative", kind::time_since_first},
	               {"frame.time_delta", kind::time_since_previous},
	               {"ip.src", kind::text},
	               {"ipv6.src", kind::text},
	               {"tcp.srcport", kind::text},
	               {"tcp.flags", kind::text},
	               {"http.request.method", kind::text},
	               {"http.request_in", kind::frames},
	               {"http.response_in", kind::frames},
	               {"tcp.reassembled_in", kind::frames},
	               {"dns.qry.name", kind::text},
	               {"dns.response_in", kind::frames},
	               {"_ws.col.Info", kind::summary}};
	opts.tshark_options = {"-E", "occurrence=f", "-o", "tcp.check_checksum:FALSE"};
	opts.environment = pvpcap::tshark_environment();
	opts.work_directory = work;
	opts.smallest_split = 0;
	opts.smallest_part = 1;
	return opts;
}

using lines = std::map<uint64_t, std::pair<uint32_t, std::string>>;

lines extract(const std::string& capture, pvpcap::extraction::options opts, size_t* parts = nullptr)
{
	pvpcap::extraction extraction(capture, opts);
	if (parts) {
		*parts = extraction.parts();
	}
	lines result;
	pvpcap::extraction::line l;
	while (extraction.next(l)) {
		result[l.frame] = {l.stream, std::string(l.fields)};
	}
	PV_VALID(extraction.damage(), std::string());
	return result;
}

std::vector<std::string> fields_of(const std::string& line)
{
	std::vector<std::string> fields(1);
	for (size_t i = 0; i < line.size(); i++) {
		if (line[i] == '\\' and i + 1 < line.size()) {
			fields.back() += line.substr(i, 2);
			i++;
		} else if (line[i] == pvpcap::SEPARATOR[0]) {
			fields.emplace_back();
		} else {
			fields.back() += line[i];
		}
	}
	return fields;
}

void check_same(const char* what, const lines& reference, const lines& tested)
{
	std::cout << what << ": " << tested.size() << " lines" << std::endl;
	PV_VALID(tested.size(), reference.size());
	for (const auto& [frame, line] : reference) {
		const auto it = tested.find(frame);
		PV_ASSERT_VALID(it != tested.end(), "frame", frame);
		PV_ASSERT_VALID(it->second == line, "frame", frame, "split", it->second.second, "whole",
		                line.second);
	}
}

} // namespace

int main()
{
	// Somewhere else than UTC: tshark writes local times, which the import reads as UTC.
	PVCore::setenv("TZ", "CET-1CEST,M3.5.0,M10.5.0/3", 1);
	pvtest::init_ctxt();
	if (pvpcap::ws_get_version().empty()) {
		std::cerr << "tshark is not available, skipping pcap integration test" << std::endl;
		return 77;
	}

	QTemporaryDir dir;
	PV_ASSERT_VALID(dir.isValid(), "a temporary directory", "could not be made");
	const std::string work = dir.filePath("work").toStdString();

	for (const uint32_t link : {ETHERNET, LINUX_SLL}) {
		std::cout << "link type " << link << std::endl;
		const std::string capture =
		    dir.filePath(QString("capture-%1.pcap").arg(link)).toStdString();
		write_capture(capture, link);

		pvpcap::extraction::options whole = profile_options(work);
		whole.workers = 1;
		const lines reference = extract(capture, whole);

		// Every frame is decoded: on a link of another type than Ethernet too.
		const std::vector<std::string> first = fields_of(reference.at(1).second);
		PV_VALID(first[0], std::string("1"));
		PV_VALID(first[2], std::string("2023-11-14T22:13:20.000000000+0000"));
		PV_ASSERT_VALID(not first[5].empty() or not first[6].empty(), "the address of frame 1",
		                "is empty");
		// A response names the frame of its request, which a single pass knows of.
		size_t requests = 0;
		size_t responses = 0;
		for (const auto& [frame, line] : reference) {
			const std::vector<std::string> fields = fields_of(line.second);
			requests += not fields[9].empty();
			responses += not fields[10].empty();
		}
		PV_VALID(requests, size_t(48));
		PV_VALID(responses, size_t(48));

		size_t parts = 0;
		pvpcap::extraction::options fed = profile_options(work);
		fed.workers = 4;
		check_same("fed through pipes", reference, extract(capture, fed, &parts));
		PV_VALID(parts, size_t(4));

		pvpcap::extraction::options written = profile_options(work);
		written.workers = 3;
		written.memory = 1; // too little to feed them all at once
		written.smallest_part = QFileInfo(QString::fromStdString(capture)).size() / 5;
		check_same("written to files", reference, extract(capture, written, &parts));
		PV_ASSERT_VALID(parts >= 3, "parts", parts);

		// A single pass can't name the frames after: the response to a request, the frame a
		// PDU is reassembled in.
		const auto count = [](const lines& extracted, size_t field) {
			return std::count_if(extracted.begin(), extracted.end(), [&](const auto& line) {
				return not fields_of(line.second.second)[field].empty();
			});
		};
		PV_VALID(count(reference, 11), std::ptrdiff_t(0));
		PV_VALID(count(reference, 12), std::ptrdiff_t(0));
		PV_VALID(count(reference, 14), std::ptrdiff_t(0));

		pvpcap::extraction::options whole_twice = profile_options(work);
		whole_twice.workers = 1;
		whole_twice.two_pass = true;
		const lines twice = extract(capture, whole_twice);
		PV_VALID(count(twice, 11), std::ptrdiff_t(48));
		PV_VALID(count(twice, 12), std::ptrdiff_t(48));
		PV_VALID(count(twice, 14), std::ptrdiff_t(24));

		pvpcap::extraction::options split_twice = profile_options(work);
		split_twice.workers = 4;
		split_twice.two_pass = true;
		check_same("read twice", twice, extract(capture, split_twice, &parts));
		PV_VALID(parts, size_t(4));

		pvpcap::extraction::options written_twice = profile_options(work);
		written_twice.workers = 3;
		written_twice.two_pass = true;
		written_twice.memory = 1;
		written_twice.smallest_part = QFileInfo(QString::fromStdString(capture)).size() / 5;
		check_same("read twice, more parts than tshark at once", twice,
		           extract(capture, written_twice, &parts));
		PV_ASSERT_VALID(parts >= 3, "parts", parts);

		// What the display filter leaves keeps its frame numbers.
		pvpcap::extraction::options filtered = profile_options(work);
		filtered.workers = 4;
		filtered.tshark_options.insert(filtered.tshark_options.end(), {"-Y", "http.request"});
		const lines kept = extract(capture, filtered);
		lines expected;
		for (const auto& [frame, line] : reference) {
			if (not fields_of(line.second)[9].empty()) {
				expected[frame] = line;
			}
		}
		// The time since the previous frame is about the frame captured before, shown or not.
		check_same("filtered", expected, kept);
	}

	std::cout << "only what tshark writes of every frame" << std::endl;
	{
		// Which it is not asked for twice: the length of the frame then ends the line.
		const std::string capture = dir.filePath("capture-1.pcap").toStdString();
		pvpcap::extraction::options opts = profile_options(work);
		opts.fields = {{"frame.len", kind::text}, {"frame.number", kind::text}};
		opts.workers = 4;
		const lines extracted = extract(capture, opts);
		PV_VALID(extracted.size(), size_t(528));
		for (const auto& [frame, line] : extracted) {
			const std::vector<std::string> fields = fields_of(line.second);
			PV_VALID(fields.size(), size_t(2));
			PV_ASSERT_VALID(not fields[0].empty(), "the length of frame", frame);
			PV_VALID(fields[1], std::to_string(frame));
		}
	}

	std::cout << "a field tshark does not know" << std::endl;
	{
		const std::string capture = dir.filePath("capture-1.pcap").toStdString();
		pvpcap::extraction::options opts = profile_options(work);
		opts.fields = {{"no.such.field", kind::text}};
		std::string said;
		try {
			extract(capture, opts);
		} catch (const pvpcap::extraction_error& e) {
			said = e.what();
		}
		std::cout << "tshark said: " << said << std::endl;
		PV_ASSERT_VALID(said.find("no.such.field") != std::string::npos, "what tshark said", said);
	}

	std::cout << "canceled" << std::endl;
	{
		const std::string capture = dir.filePath("capture-1.pcap").toStdString();
		pvpcap::extraction::options opts = profile_options(work);
		opts.workers = 4;
		{
			pvpcap::extraction extraction(capture, opts);
			extraction.cancel();
			pvpcap::extraction::line l;
			PV_VALID(extraction.next(l), false);
		}
		// Nothing is left behind.
		PV_VALID(QDir(QString::fromStdString(work))
		             .entryList(QDir::NoDotAndDotDot | QDir::AllEntries)
		             .size(),
		         qsizetype(0));
	}

	return 0;
}
