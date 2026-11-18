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

// An investigation of a capture opens without its NRAW.
//
// The NRAW of an investigation is a cache: once it is gone, the source is extracted again
// from its inputs. The inputs of a capture used to be the parts the import split it into,
// removed as soon as read: reopened without its NRAW, the investigation of a capture could
// not be. They are now the capture itself, with what tshark is asked of it, and the rows
// come back the same, in the same order, which the layers and selections saved rely on.

#include "../../libpvkernel/plugins/common/pcap/PVPcapDescription.h"
#include "../../libpvkernel/plugins/common/pcap/libpvpcap/include/libpvpcap/profileformat.h"
#include "../../libpvkernel/plugins/common/pcap/libpvpcap/include/libpvpcap/ws.h"
#include "../../libpvkernel/tests/rush/pcap_captures.h"

#include <squey/PVRoot.h>
#include <squey/PVSource.h>

#include <pvkernel/core/PVDirectory.h>
#include <pvkernel/core/PVSerializeArchiveZip.h>
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/rush/PVSourceCreator.h>

#include <rapidjson/document.h>

#include <QDir>

#include <iostream>
#include <vector>

#include "common.h"

using namespace pcap_captures;

namespace
{

const char* PROFILE = R"json({
  "options": {
    "source": false, "destination": false, "protocol": true, "info": true,
    "filters": "", "header": false, "aggregator": "|", "occurrence": "f"
  },
  "children": [{"fields": [
    {"name": "", "filter_name": "frame.time", "type": "FT_ABSOLUTE_TIME", "description": "", "select": true},
    {"name": "", "filter_name": "ip.src", "type": "FT_IPv4", "description": "", "select": true},
    {"name": "", "filter_name": "tcp.srcport", "type": "FT_UINT16", "description": "", "select": true},
    {"name": "", "filter_name": "tcp.flags", "type": "FT_UINT16", "description": "", "select": true}
  ], "children": []}]
})json";

std::vector<std::string> rows_of(const PVRush::PVNraw& nraw)
{
	std::vector<std::string> rows;
	for (PVRow row = 0; row < nraw.row_count(); row++) {
		std::string text;
		for (PVCol col(0); col < PVCol(nraw.column_count()); col++) {
			text += nraw.at_string(row, col) + "|";
		}
		rows.push_back(text);
	}
	return rows;
}

} // namespace

int main()
{
	pvtest::TestEnv env;
	if (pvpcap::ws_get_version().empty()) {
		std::cerr << "tshark is not available, skipping pcap integration test" << std::endl;
		return 77;
	}

	const QString directory = PVRush::PVNrawCacheManager::nraw_dir();
	QDir().mkpath(directory);
	const QString capture = directory + "/connections.pcap";
	std::vector<frame> frames;
	for (int c = 0; c < 50; c++) {
		const ipv4_address client{10, 0, 0, static_cast<uint8_t>(c + 1)};
		const ipv4_address server{192, 168, 1, 1};
		const uint16_t port = static_cast<uint16_t>(30000 + c);
		frames.push_back({1700000000, uint32_t(c * 10),
		                  ethernet(0x0800, ipv4(client, server, TCP, tcp(port, 80, 1, 0, SYN)))});
		frames.push_back(
		    {1700000000, uint32_t(c * 10 + 5),
		     ethernet(0x0800, ipv4(server, client, TCP, tcp(80, port, 9, 2, SYN | ACK)))});
	}
	write_pcap(capture.toStdString(), 1, frames);

	rapidjson::Document profile;
	profile.Parse(PROFILE);
	PV_VALID(profile.HasParseError(), false);
	const PVRush::PVFormat format(pvpcap::get_format(profile, 1).documentElement());
	PVRush::PVInputType::list_inputs inputs;
	inputs << PVRush::PVInputDescription_p(
	    new PVRush::PVPcapDescription(capture, pvpcap::ws_extraction_options(profile),
	                                  std::make_shared<PVRush::PVPcapImport>(1), 0, false));

	Squey::PVSource& source = env.add_source(
	    inputs, LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("pcap"), format);
	env.compute_mappings();
	env.compute_scalings();
	env.compute_views();
	const std::vector<std::string> imported = rows_of(source.get_rushnraw());
	PV_VALID(imported.size(), size_t(100));
	const std::string nraw_directory = source.get_rushnraw().dir();

	const std::string investigation = directory.toStdString() + "/capture.pvi";
	{
		PVCore::PVSerializeArchiveZip archive(QString::fromStdString(investigation),
		                                      PVCore::PVSerializeArchive::write,
		                                      SQUEY_ARCHIVES_VERSION, false);
		env.root.save_to_file(archive);
	}

	// What a cleaned cache leaves of the investigation. Windows deletes no file still open: the
	// source closes first, as it does when Squey exits.
	env.root.remove_all_children();
	PVCore::PVDirectory::remove_rec(QString::fromStdString(nraw_directory));
	PV_VALID(QDir(QString::fromStdString(nraw_directory)).exists(), false);

	Squey::PVRoot root;
	{
		PVCore::PVSerializeArchiveZip archive(QString::fromStdString(investigation),
		                                      PVCore::PVSerializeArchive::read,
		                                      SQUEY_ARCHIVES_VERSION, false);
		root.load_from_archive(archive);
	}
	const auto sources = root.get_children<Squey::PVSource>();
	PV_VALID(sources.size(), size_t(1));
	const std::vector<std::string> reloaded = rows_of(sources.front()->get_rushnraw());
	PV_VALID(reloaded.size(), imported.size());
	for (size_t row = 0; row < imported.size(); row++) {
		PV_ASSERT_VALID(reloaded[row] == imported[row], "row", row, "imported", imported[row],
		                "reloaded", reloaded[row]);
	}
	std::cout << reloaded.size() << " rows extracted again, the same" << std::endl;

	return 0;
}
