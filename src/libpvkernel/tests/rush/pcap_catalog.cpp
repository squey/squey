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

// What helps choose the fields of a pcap import.
//
// tshark lists the protocols and the fields it knows, which a search finds by their filter
// names, names and descriptions, the likeliest first. The first frames of a capture tell
// which protocols and fields it holds, with an example of their values, and about how many
// frames the whole capture has. And tshark tells what is wrong with a display filter, and
// where.

#include <pvkernel/core/PVConfig.h>
#include <pvkernel/core/squey_assert.h>

#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap/catalog.h"
#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap/extraction.h"
#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap/sample.h"
#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap/ws.h"
#include "common.h"

#include <QDir>
#include <QFile>
#include <QTemporaryDir>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>

int main(int argc, char** argv)
{
	if (argc < 3) {
		std::cerr << "usage: " << argv[0] << " <pvconfig.ini> <capture of HTTP>" << std::endl;
		return 1;
	}
	// The configuration of the user, which PVConfig reads, in the directory of its own CTest
	// sets: the test does not count on another one to have made it.
	QDir().mkpath(PVCore::PVConfig::user_dir());
	QFile::remove(PVCore::PVConfig::user_path());
	PV_VALID(QFile::copy(argv[1], PVCore::PVConfig::user_path()), true);
	pvtest::init_ctxt();
	if (pvpcap::ws_get_version().empty()) {
		std::cerr << "tshark is not available, skipping" << std::endl;
		return 77;
	}
	const std::string capture = argv[2];

	std::cout << "the fields tshark knows" << std::endl;
	const auto start = std::chrono::steady_clock::now();
	const std::shared_ptr<const pvpcap::field_catalog> catalog = pvpcap::field_catalog::load();
	std::cout << catalog->protocols().size() << " protocols, " << catalog->fields().size()
	          << " fields, read in "
	          << std::chrono::duration_cast<std::chrono::milliseconds>(
	                 std::chrono::steady_clock::now() - start)
	                 .count()
	          << " ms" << std::endl;
	PV_ASSERT_VALID(catalog->fields().size() > 10000, "fields", catalog->fields().size());
	const int64_t host = catalog->find_field("http.host");
	PV_ASSERT_VALID(host >= 0, "http.host", "not found");
	const pvpcap::field_catalog::field& host_field = catalog->fields()[host];
	PV_VALID(std::string(host_field.type), std::string("FT_STRING"));
	PV_VALID(std::string(catalog->protocols()[host_field.protocol].filter_name),
	         std::string("http"));
	const auto& http_fields = catalog->protocols()[catalog->find_protocol("http")].fields;
	PV_ASSERT_VALID(std::find(http_fields.begin(), http_fields.end(), uint32_t(host)) !=
	                    http_fields.end(),
	                "http.host", "is not a field of http");
	// The columns of Wireshark, under the names profiles give them, and only those.
	for (const char* column :
	     {"_ws.col.Source", "_ws.col.Destination", "_ws.col.Protocol", "_ws.col.Info"}) {
		PV_ASSERT_VALID(catalog->find_field(column) >= 0, "column", column);
	}
	for (const char* alias : {"_ws.col.info", "_ws.col.def_src", "_ws.col.number"}) {
		PV_VALID(catalog->find_field(alias), int64_t(-1));
	}

	std::cout << "searched" << std::endl;
	const auto found = [&](const char* query, size_t most) {
		std::vector<std::string> names;
		for (const uint32_t i : catalog->search(query, most)) {
			names.emplace_back(catalog->fields()[i].filter_name);
		}
		return names;
	};
	const auto among = [&](const char* name, const char* query, size_t most) {
		const std::vector<std::string> names = found(query, most);
		PV_ASSERT_VALID(std::find(names.begin(), names.end(), name) != names.end(), "field", name,
		                "not among the first results of", query);
	};
	PV_VALID(found("ip.src", 1).at(0), std::string("ip.src"));
	PV_VALID(found("  IP.SRC ", 1).at(0), std::string("ip.src"));
	PV_VALID(found("dns.qry.name", 1).at(0), std::string("dns.qry.name"));
	among("http.user_agent", "user agent", 3);
	among("http.host", "http host", 3);
	among("_ws.col.Info", "info", 5);
	PV_VALID(found("no such field anywhere", 10).empty(), true);
	PV_VALID(found("", 10).empty(), true);
	PV_VALID(found("tcp", 25).size(), size_t(25));
	// Fields seen in the captures first.
	const int64_t sni = catalog->find_field("tls.handshake.extensions_server_name");
	const std::vector<uint32_t> boosted =
	    catalog->search("server name", 1, [&](uint32_t i) { return int64_t(i) == sni ? 5000 : 0; });
	PV_VALID(int64_t(boosted.at(0)), sni);

	std::cout << "the start of a capture" << std::endl;
	const pvpcap::capture_overview whole = pvpcap::overview_capture(capture, 1000);
	PV_VALID(whole.format, std::string("pcap"));
	PV_VALID(whole.compressed, false);
	PV_VALID(whole.frames, uint64_t(43));
	PV_VALID(whole.whole, true);
	PV_VALID(whole.estimated_frames, uint64_t(43));
	PV_ASSERT_VALID(whole.first_timestamp > 0 and whole.last_timestamp > whole.first_timestamp,
	                "first", whole.first_timestamp, "last", whole.last_timestamp);
	const pvpcap::capture_overview start_only = pvpcap::overview_capture(capture, 10);
	PV_VALID(start_only.frames, uint64_t(10));
	PV_VALID(start_only.whole, false);
	PV_ASSERT_VALID(start_only.estimated_frames > 20 and start_only.estimated_frames < 90,
	                "estimated", start_only.estimated_frames);

	QTemporaryDir dir;
	const std::string text = dir.filePath("not a capture.txt").toStdString();
	std::ofstream(text) << "not a capture";
	const pvpcap::capture_overview unknown = pvpcap::overview_capture(text, 10);
	PV_VALID(unknown.format, std::string());
	PV_VALID(unknown.size, uint64_t(13));

	std::cout << "what its first frames hold" << std::endl;
	std::atomic<bool> canceled = false;
	const pvpcap::capture_sample sample = pvpcap::sample_capture(capture, 20, {}, canceled);
	PV_VALID(sample.frames, uint64_t(20));
	const auto protocol = [&](const char* name) {
		const auto it = std::find_if(sample.protocols.begin(), sample.protocols.end(),
		                             [&](const auto& p) { return p.name == name; });
		PV_ASSERT_VALID(it != sample.protocols.end(), "protocol", name);
		return *it;
	};
	PV_VALID(protocol("eth").frames, uint64_t(20));
	PV_VALID(protocol("eth").depth, uint32_t(0));
	PV_VALID(protocol("tcp").depth, uint32_t(3));
	PV_VALID(protocol("http").frames, uint64_t(2));
	PV_VALID(sample.fields.at("ip.src").frames, uint64_t(20));
	PV_VALID(sample.fields.at("http.request.method").frames, uint64_t(2));
	PV_VALID(sample.fields.at("http.request.method").example, std::string("GET"));
	PV_VALID(sample.fields.contains("http.response.code"), false);

	bool refused = false;
	try {
		pvpcap::sample_capture(text, 20, {}, canceled);
	} catch (const pvpcap::extraction_error& e) {
		std::cout << "tshark said: " << e.what() << std::endl;
		refused = true;
	}
	PV_VALID(refused, true);

	canceled = true;
	PV_VALID(pvpcap::sample_capture(capture, 20, {}, canceled).frames, uint64_t(0));

	std::cout << "display filters" << std::endl;
	const pvpcap::filter_check right = pvpcap::ws_check_filter("http.request and tcp.port == 80");
	PV_VALID(right.checked, true);
	PV_VALID(right.problem, std::string());
	const pvpcap::filter_check wrong = pvpcap::ws_check_filter("tcp.port == 80 and http.reqest");
	std::cout << "tshark said: " << wrong.problem << std::endl;
	PV_VALID(wrong.checked, true);
	PV_ASSERT_VALID(wrong.problem.find("http.reqest") != std::string::npos, "the problem",
	                wrong.problem);
	PV_VALID(wrong.position, size_t(19));
	PV_VALID(wrong.length, size_t(11));
	const pvpcap::filter_check unfinished = pvpcap::ws_check_filter("http.request ==");
	PV_ASSERT_VALID(not unfinished.problem.empty(), "an unfinished filter", "was accepted");

	return 0;
}
