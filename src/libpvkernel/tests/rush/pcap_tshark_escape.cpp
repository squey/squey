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

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include <rapidjson/document.h>

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/rush/PVSourceCreator.h>

#include "../../plugins/common/pcap/PVPcapDescription.h"
#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap.h"
#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap/pcap_splitter.h"
#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap/profileformat.h"
#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap/shell.h"
#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap/ws.h"

#include "common.h"

/*
 * tshark_escape_trap.pcap contains 4 crafted DNS TXT responses whose payload
 * bytes are:
 *   #1 "clean-value"     (control)
 *   #2 "pre" 0x7F "post" (the pcap plugin's own CSV field separator)
 *   #3 "line1" \n "line2" (a real newline byte)
 *   #4 "a" 0x7F "b" \n "cd" (both at once)
 *
 * tshark backslash-escapes both bytes in its -Tfields output. This test
 * drives the real pcap import pipeline (split_pcaps -> extract_csv, using the
 * exact tshark command built by ws_get_cmdline_opts()) end to end, and checks
 * that dns.txt reaches the NRAW unshifted, and that the following column
 * (ip.dst) is never contaminated by the escaped bytes.
 *
 * The escaped separator (0x7F) is unescaped back to its literal byte, but the
 * escaped newline is intentionally left as its 2-character "\n" escape form:
 * pvcop's string dictionary storage uses a raw newline byte as its own
 * on-disk record separator (pvcop::db::write_dict::save), so turning it back
 * into a literal newline would silently truncate the value one layer down.
 */
static const char* PROFILE_JSON = R"json(
{
  "options": {
    "source": false, "destination": false, "protocol": false, "info": false,
    "filters": "", "header": false, "aggregator": "|", "occurrence": "f"
  },
  "children": [
    {
      "fields": [
        {"name": "TXT", "filter_name": "dns.txt", "type": "FT_STRING", "description": "", "select": true},
        {"name": "Destination", "filter_name": "ip.dst", "type": "FT_IPv4", "description": "", "select": true}
      ],
      "children": []
    }
  ]
}
)json";

UNICODE_MAIN()
{
	if (argc <= 1) {
		std::cerr << "Usage: <pcap_file>" << std::endl;
		return 1;
	}
#ifdef _WIN32
	std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
	const std::string pcap_path = conv.to_bytes(argv[1]);
#else
	const std::string pcap_path = argv[1];
#endif

	pvtest::init_ctxt();

	// This is an integration test that shells out to tshark. On Windows/macOS
	// tshark_path() resolves into the app cache directory, which only squey.exe
	// populates at startup; the CI therefore points SQUEY_TSHARK_PATH at the
	// bundled tshark for the testsuite (see .gitlab-ci/run_testsuite.ps1). If
	// tshark still cannot be run, skip (SKIP_RETURN_CODE, see CMakeLists) rather
	// than fail.
	const std::vector<std::string> tshark_version =
	    pvpcap::execute_cmd(pvpcap::tshark_path() + " -v");
	const bool tshark_available =
	    std::any_of(tshark_version.begin(), tshark_version.end(), [](const std::string& line) {
		    return line.find("TShark") != std::string::npos or
		           line.find("Wireshark") != std::string::npos;
	    });
	if (not tshark_available) {
		std::cerr << "tshark is not available, skipping pcap integration test" << std::endl;
		return 77;
	}

	rapidjson::Document json_data;
	json_data.Parse(PROFILE_JSON);
	PV_VALID(json_data.HasParseError(), false);

	// 1) split by flow, exactly as PVPcapParamsWidget does. All 4 crafted
	//    packets share the same 5-tuple, so this yields a single chunk.
	bool canceled = false;
	const std::string output_dir =
	    PVRush::PVNrawCacheManager::nraw_dir().toStdString() + "/pcap_tshark_escape";
	pvpcap::splitted_files_t splitted =
	    pvpcap::split_pcaps({pcap_path}, output_dir, /*preserve_flows=*/true, canceled);
	PV_VALID(splitted.empty(), false);

	// 2) run the exact tshark command the production code path builds
	std::vector<std::string> cmd = pvpcap::ws_get_cmdline_opts(json_data);
	pvpcap::splitted_files_t csv_files = pvpcap::extract_csv(std::move(splitted), cmd, canceled);
	PV_VALID(csv_files.empty(), false);

	// 3) build the format and inputs, mirroring PVInputTypePcap::load_files()
	QDomDocument fmt_doc = pvpcap::get_format(json_data, /*input_pcap_count=*/1);
	PVRush::PVFormat format(fmt_doc.documentElement());

	QList<PVRush::PVInputDescription_p> list_inputs;
	size_t packets_count_offset = 0;
	size_t streams_id_offset = 0;
	for (pvpcap::splitted_file_t& sf : csv_files) {
		list_inputs << PVRush::PVInputDescription_p(new PVRush::PVPcapDescription(
		    QString::fromStdString(sf.path()), QString::fromStdString(sf.original_pcap_path()),
		    packets_count_offset, sf.packets_indexes(), streams_id_offset, sf.streams_ids(),
		    /*multi_inputs=*/false));
		packets_count_offset += sf.packets_count();
		streams_id_offset += sf.streams_ids_count();
	}

	// 4) run the real "pcap" PVSourceCreator through the extractor, up to the NRAW
	PVRush::PVSourceCreator_p sc = LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("pcap");
	PVRush::PVNraw nraw;
	PVRush::PVNrawOutput output(nraw);
	PVRush::PVExtractor extractor(format, output, sc, list_inputs);
	PVRush::PVControllerJob_p job = extractor.process_from_agg_idxes(0, IMPORT_PIPELINE_ROW_COUNT_LIMIT);
	job->wait_end();

	PV_VALID(nraw.row_count(), (PVRow)4);

	// locate columns by name rather than assuming a fixed layout
	int col_frame = -1;
	int col_txt = -1;
	int col_dst = -1;
	const QList<PVRush::PVAxisFormat>& axes = format.get_axes();
	for (int i = 0; i < axes.size(); i++) {
		const QString& name = axes[i].get_name();
		if (name == "frame.global_number") {
			col_frame = i;
		} else if (name == "dns.txt") {
			col_txt = i;
		} else if (name == "ip.dst") {
			col_dst = i;
		}
	}
	PV_VALID(col_frame >= 0 and col_txt >= 0 and col_dst >= 0, true);

	static const std::unordered_map<std::string, std::string> expected_txt = {
	    {"1", "clean-value"},
	    {"2", "pre\x7F" "post"},
	    {"3", "line1\\nline2"},   // literal 2-char backslash-n escape, not a real newline
	    {"4", "a\x7F" "b\\ncd"},
	};

	for (PVRow row(0); row < nraw.row_count(); row++) {
		const std::string& frame_number = nraw.at_string(row, PVCol(col_frame));
		const auto it = expected_txt.find(frame_number);
		PV_VALID(it != expected_txt.end(), true);
		PV_VALID(nraw.at_string(row, PVCol(col_txt)), it->second);
		// the escaped byte in dns.txt must never shift the following column
		PV_VALID(nraw.at_string(row, PVCol(col_dst)), std::string("10.2.2.2"));
	}

	return 0;
}
