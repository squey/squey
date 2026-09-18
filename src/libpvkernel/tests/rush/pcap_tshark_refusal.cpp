//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

// A profile tshark refuses comes back in tshark's own words.
//
// tshark says plainly what it does not accept -- here a field it has never heard
// of -- on a standard error nobody used to read, and the import was left with
// empty csv files and no idea why. extract_csv() now hands back what the first
// child that failed said.

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <rapidjson/document.h>

#include <pvkernel/core/PVUtils.h>
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNrawCacheManager.h>

#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap.h"
#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap/pcap_splitter.h"
#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap/shell.h"
#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap/ws.h"

#include "common.h"

static const char* PROFILE_JSON = R"json(
{
  "options": {
    "source": false, "destination": false, "protocol": false, "info": false,
    "filters": "", "header": false, "aggregator": "|", "occurrence": "f"
  },
  "children": [
    {
      "fields": [
        {"name": "Nothing", "filter_name": "no.such.field", "type": "FT_STRING", "description": "", "select": true}
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
	const std::string pcap_path = PVCore::wide_to_utf8(argv[1]);
#else
	const std::string pcap_path = argv[1];
#endif

	pvtest::init_ctxt();

	// Skipped where tshark cannot be run, as Trush_pcap_tshark_escape is.
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

	bool canceled = false;
	const std::string output_dir =
	    PVRush::PVNrawCacheManager::nraw_dir().toStdString() + "/pcap_tshark_refusal";
	pvpcap::splitted_files_t splitted =
	    pvpcap::split_pcaps({pcap_path}, output_dir, /*preserve_flows=*/true, canceled);
	PV_VALID(splitted.empty(), false);

	std::string trouble;
	pvpcap::extract_csv(std::move(splitted), pvpcap::ws_get_cmdline_opts(json_data), canceled, {},
	                    {}, &trouble);
	std::cout << "tshark said: " << trouble << std::endl;
	PV_ASSERT_VALID(trouble.find("no.such.field") != std::string::npos, "what tshark said",
	                trouble);

	return 0;
}
