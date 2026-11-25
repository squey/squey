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

// A profile says which fields of the frames of captures make the columns of an import, and
// how tshark reads the captures.
//
// Earlier versions wrote the fields selected in a tree of protocols: they read as the
// columns, in the order of the tree, each field once, then the columns of Wireshark chosen
// in the options. What this version writes reads back the same, and earlier versions read it
// as a profile of no field rather than failing on it.
//
// The profiles of the user are files named after them: a name that can't name a file is
// refused, and a new profile is given a name no other has. What was changed of profiles and
// not saved yet is kept until it is.

#include <pvkernel/core/PVConfig.h>
#include <pvkernel/core/PVUtils.h>
#include <pvkernel/core/squey_assert.h>

#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap/profile.h"
#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap/shell.h"
#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap/ws.h"
#include <QCoreApplication>
#include <QDir>
#include <QFile>

#include <iostream>

namespace
{

// Two protocols, a field selected in both, and the Info column of Wireshark under the name
// tshark gives it as well as in the options.
const char LEGACY[] = R"json({
  "read_only": false,
  "options": {"source": true, "destination": false, "protocol": false, "info": true,
              "filters": "dns", "occurrence": "a", "aggregator": ",",
              "nameres.network_name": true, "ip.defragment": false,
              "wireshark_profile": "Mine"},
  "children": [{"filter_name": "frame", "fields": [
      {"filter_name": "frame.time", "type": "FT_ABSOLUTE_TIME", "select": true},
      {"filter_name": "frame.len", "type": "FT_UINT32", "select": false}],
    "children": [{"filter_name": "ip", "fields": [
        {"filter_name": "ip.src", "type": "FT_IPv4", "select": true},
        {"filter_name": "frame.time", "type": "FT_ABSOLUTE_TIME", "select": true},
        {"filter_name": "_ws.col.info", "type": "FT_STRING", "select": true}],
      "children": []}]}]
})json";

std::string joined(const std::vector<std::string>& words)
{
	std::string text;
	for (const std::string& word : words) {
		text += (text.empty() ? "" : " ") + word;
	}
	return text;
}

std::string names(const pvpcap::profile& p)
{
	std::vector<std::string> columns;
	for (const pvpcap::profile_field& f : p.columns) {
		columns.push_back(f.name);
	}
	return joined(columns);
}

} // namespace

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "usage: " << argv[0] << " <directory of the profiles Squey ships>"
		          << std::endl;
		return 1;
	}
	QCoreApplication application(argc, argv);
	PVCore::setenv("SQUEY_PCAP_PROFILES_PATH", argv[1], 1);
	QDir(QString::fromStdString(pvpcap::get_user_conf_dir())).removeRecursively();

	std::cout << "a profile of an earlier version" << std::endl;
	rapidjson::Document legacy_json;
	legacy_json.Parse(LEGACY);
	const pvpcap::profile legacy = pvpcap::profile_from_json(legacy_json);
	PV_VALID(names(legacy), std::string("frame.time ip.src _ws.col.Info _ws.col.Source"));
	PV_VALID(legacy.columns.front().type, std::string("FT_ABSOLUTE_TIME"));
	PV_VALID(legacy.filter, std::string("dns"));
	PV_VALID(legacy.occurrence, 'a');
	PV_VALID(legacy.aggregator, std::string(","));
	PV_VALID(legacy.reassemble_tcp, true);
	PV_VALID(legacy.defragment_ip, false);
	PV_VALID(legacy.resolve_names, true);
	PV_VALID(legacy.names_from_captures, false);
	PV_VALID(legacy.wireshark_profile, std::string("Mine"));

	// What tshark was asked before, except for the columns of Wireshark, asked once now.
	const pvpcap::extraction::options asked = pvpcap::ws_extraction_options(legacy);
	PV_VALID(asked.fields.size(), size_t(4));
	PV_VALID(asked.fields[2].type == pvpcap::extraction::field::kind::summary, true);
	PV_VALID(joined({asked.tshark_options.begin(), asked.tshark_options.begin() + 10}),
	         std::string("-Y dns -E occurrence=a -E aggregator=, -o ip.defragment:FALSE -N n"));

	std::cout << "a profile with missing options" << std::endl;
	{
		rapidjson::Document json;
		json.Parse(R"({"children": []})");
		const pvpcap::profile p = pvpcap::profile_from_json(json);
		// As earlier versions read them.
		PV_VALID(p.occurrence, 'f');
		PV_VALID(p.reassemble_tcp, true);
		PV_VALID(p.columns.empty(), true);
		bool refused = false;
		try {
			json.Parse("[]");
			pvpcap::profile_from_json(json);
		} catch (const pvpcap::profile_error&) {
			refused = true;
		}
		PV_VALID(refused, true);
	}

	std::cout << "written and read back" << std::endl;
	const std::string path = pvpcap::get_user_profile_path("Legacy");
	pvpcap::write_profile(legacy, path);
	PV_ASSERT_VALID(pvpcap::read_profile(path) == legacy, "the profile read back from", path);
	{
		// As earlier versions read it: no field, rather than a crash.
		QFile file(QString::fromStdString(path));
		PV_VALID(file.open(QIODevice::ReadOnly), true);
		rapidjson::Document json;
		json.Parse(file.readAll().constData());
		PV_VALID(json["children"].IsArray() and json["children"].Empty(), true);
		PV_VALID(json["read_only"].IsBool(), true);
		for (const char* option : {"source", "destination", "protocol", "info", "header"}) {
			PV_ASSERT_VALID(json["options"][option].IsBool(), "option", option);
		}
		PV_VALID(std::string(json["options"]["filters"].GetString()), std::string("dns"));
	}
	bool unreadable = false;
	try {
		pvpcap::read_profile(pvpcap::get_user_profile_path("Nothing there"));
	} catch (const pvpcap::profile_error&) {
		unreadable = true;
	}
	PV_VALID(unreadable, true);

	std::cout << "the profiles" << std::endl;
	const std::vector<pvpcap::profile_location> profiles = pvpcap::list_profiles();
	const size_t builtin =
	    std::count_if(profiles.begin(), profiles.end(), [](const auto& p) { return p.builtin; });
	PV_ASSERT_VALID(builtin >= 1, "the profiles in", argv[1]);
	PV_ASSERT_VALID(profiles.back() == (pvpcap::profile_location{false, "Legacy"}),
	                "the last profile", profiles.back().name);
	for (const pvpcap::profile_location& location : profiles) {
		const pvpcap::profile p = pvpcap::read_profile(location.path());
		PV_ASSERT_VALID(not p.columns.empty(), "the columns of", location.name);
	}

	std::cout << "names" << std::endl;
	for (const char* wrong : {"", " Mine", "Mine ", ".hidden", "trailing.", "a/b", "a\\b", "what?",
	                          "CON", "lpt1.json", "tab\there"}) {
		PV_ASSERT_VALID(not pvpcap::profile_name_problem(wrong).empty(), "the name", wrong);
	}
	for (const char* right : {"Mine", "HTTP (2)", "Requests – été", "a.b"}) {
		PV_ASSERT_VALID(pvpcap::profile_name_problem(right).empty(), "the name", right, "said",
		                pvpcap::profile_name_problem(right));
	}
	PV_VALID(pvpcap::user_profile_exists("legacy"), true);
	PV_VALID(pvpcap::free_profile_name("Legacy"), std::string("Legacy 2"));
	PV_VALID(pvpcap::free_profile_name("New"), std::string("New"));

	std::cout << "drafts" << std::endl;
	PV_VALID(pvpcap::read_drafts().empty(), true);
	pvpcap::profile changed = legacy;
	changed.columns.pop_back();
	changed.filter.clear();
	const pvpcap::profile_drafts drafts = {{{true, "HTTP"}, changed}, {{false, "Legacy"}, legacy}};
	pvpcap::write_drafts(drafts);
	const pvpcap::profile_drafts read = pvpcap::read_drafts();
	PV_VALID(read.size(), size_t(2));
	PV_ASSERT_VALID(read == drafts, "the drafts", "differ once read back");
	pvpcap::write_drafts({});
	PV_VALID(pvpcap::read_drafts().empty(), true);

	return 0;
}
