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

// Captures import as rows, one per frame, in the order of the frames.
//
// Two captures imported together number their frames and their conversations one after the
// other, even when their files have the same name, which used to make the second overwrite
// the parts of the first. The fields tshark writes in hexadecimal or as booleans, and the
// times, come out as values rather than as invalid ones, and so does frame.len, which tshark
// only writes once however many times it is asked. And when tshark refuses the profile, the
// import fails with what it said.

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVInput.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/rush/PVSourceCreator.h>

#include <rapidjson/document.h>

#include "../../plugins/common/pcap/PVPcapDescription.h"
#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap.h"
#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap/profileformat.h"
#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap/ws.h"
#include "pcap_captures.h"
#include "common.h"

#include <QDir>
#include <QTemporaryDir>

#include <iostream>
#include <map>

using namespace pcap_captures;

namespace
{

std::string profile(const std::string& extra_field = {})
{
	std::string fields = R"json(
        {"name": "", "filter_name": "frame.time", "type": "FT_ABSOLUTE_TIME", "description": "", "select": true},
        {"name": "", "filter_name": "frame.len", "type": "FT_UINT32", "description": "", "select": true},
        {"name": "", "filter_name": "ip.src", "type": "FT_IPv4", "description": "", "select": true},
        {"name": "", "filter_name": "tcp.srcport", "type": "FT_UINT16", "description": "", "select": true},
        {"name": "", "filter_name": "tcp.flags", "type": "FT_UINT16", "description": "", "select": true},
        {"name": "", "filter_name": "tcp.flags.syn", "type": "FT_BOOLEAN", "description": "", "select": true})json";
	if (not extra_field.empty()) {
		fields += R"json(, {"name": "", "filter_name": ")json" + extra_field +
		          R"json(", "type": "FT_STRING", "description": "", "select": true})json";
	}
	return R"json({
  "options": {
    "source": false, "destination": false, "protocol": false, "info": false,
    "filters": "", "header": false, "aggregator": "|", "occurrence": "f"
  },
  "children": [{"fields": [)json" +
	       fields + R"json(], "children": []}]
})json";
}

//! A handshake on each of `count` connections, in turn, client first.
void write_capture(const std::string& path, uint8_t client_byte, int count)
{
	std::vector<frame> frames;
	uint32_t t = 0;
	for (int c = 0; c < count; c++) {
		const ipv4_address client{10, 0, client_byte, static_cast<uint8_t>(c + 1)};
		const ipv4_address server{192, 168, 1, 1};
		const uint16_t port = static_cast<uint16_t>(40000 + c);
		frames.push_back({1700000000, t++,
		                  ethernet(0x0800, ipv4(client, server, TCP, tcp(port, 80, 1, 0, SYN)))});
		frames.push_back(
		    {1700000000, t++,
		     ethernet(0x0800, ipv4(server, client, TCP, tcp(80, port, 9, 2, SYN | ACK)))});
	}
	write_pcap(path, 1, frames);
}

struct imported {
	PVRush::PVNraw nraw;
	std::map<std::string, PVCol> columns;
};

void import(imported& result, const std::string& profile_json, const QStringList& captures)
{
	rapidjson::Document json;
	json.Parse(profile_json.c_str());
	PV_VALID(json.HasParseError(), false);

	const QDomDocument format_document = pvpcap::get_format(json, captures.size());
	PVRush::PVFormat format(format_document.documentElement());
	const pvpcap::extraction::options extraction = pvpcap::ws_extraction_options(json);
	const auto shared = std::make_shared<PVRush::PVPcapImport>(captures.size());
	QList<PVRush::PVInputDescription_p> inputs;
	for (qsizetype i = 0; i < captures.size(); i++) {
		inputs << PVRush::PVInputDescription_p(
		    new PVRush::PVPcapDescription(captures[i], extraction, shared, i, captures.size() > 1));
	}

	const QList<PVRush::PVAxisFormat>& axes = format.get_axes();
	for (int i = 0; i < axes.size(); i++) {
		result.columns[axes[i].get_name().toStdString()] = PVCol(i);
	}

	PVRush::PVSourceCreator_p creator =
	    LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("pcap");
	PVRush::PVNrawOutput output(result.nraw);
	PVRush::PVExtractor extractor(format, output, creator, inputs);
	PVRush::PVControllerJob_p job =
	    extractor.process_from_agg_idxes(0, IMPORT_PIPELINE_ROW_COUNT_LIMIT);
	job->wait_end();
}

} // namespace

int main()
{
	pvtest::init_ctxt();
	if (pvpcap::ws_get_version().empty()) {
		std::cerr << "tshark is not available, skipping pcap integration test" << std::endl;
		return 77;
	}

	QTemporaryDir dir;
	PV_ASSERT_VALID(dir.isValid(), "a temporary directory", "could not be made");
	// Two captures of the same name, the second with more connections.
	QDir().mkpath(dir.filePath("a"));
	QDir().mkpath(dir.filePath("b"));
	const QString first = dir.filePath("a/capture.pcap");
	const QString second = dir.filePath("b/capture.pcap");
	write_capture(first.toStdString(), 0, 30);
	write_capture(second.toStdString(), 1, 40);

	std::cout << "two captures of the same name" << std::endl;
	{
		imported result;
		import(result, profile(), {first, second});
		const PVRush::PVNraw& nraw = result.nraw;
		PV_VALID(nraw.row_count(), PVRow(140));
		const PVCol frame = result.columns.at("frame.global_number");
		const PVCol stream = result.columns.at("stream_id");
		const PVCol source = result.columns.at("ip.src");
		const PVCol flags = result.columns.at("tcp.flags");
		const PVCol syn = result.columns.at("tcp.flags.syn");
		const PVCol time = result.columns.at("frame.time");
		// Which the extraction asks tshark for already, and which tshark only writes once.
		const PVCol length = result.columns.at("frame.len");
		for (PVRow row = 0; row < nraw.row_count(); row++) {
			// In the order of the frames, the second capture after the first.
			PV_VALID(nraw.at_string(row, frame), std::to_string(row + 1));
			// Both directions of a connection are one conversation.
			PV_VALID(nraw.at_string(row, stream), std::to_string(row / 2));
			const std::string client = row < 60 ? "10.0.0." : "10.0.1.";
			const std::string connection = std::to_string((row < 60 ? row : row - 60) / 2 + 1);
			PV_VALID(nraw.at_string(row, source),
			         row % 2 == 0 ? client + connection : std::string("192.168.1.1"));
			PV_VALID(nraw.at_string(row, flags), std::string(row % 2 == 0 ? "0x2" : "0x12"));
			PV_VALID(nraw.at_string(row, syn), std::string("True"));
			PV_ASSERT_VALID(nraw.column(time).is_valid(row), "the time of row", row);
			PV_VALID(nraw.at_string(row, length), std::string("54"));
		}
	}

	std::cout << "a field tshark does not know" << std::endl;
	{
		imported result;
		std::string said;
		try {
			import(result, profile("no.such.field"), {first});
		} catch (const PVRush::PVInputException& e) {
			said = e.what();
		}
		std::cout << "the import said: " << said << std::endl;
		PV_ASSERT_VALID(said.find("no.such.field") != std::string::npos, "what the import said",
		                said);
	}

	return 0;
}
