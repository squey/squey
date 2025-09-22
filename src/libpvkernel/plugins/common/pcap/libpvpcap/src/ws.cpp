//
// MIT License
//
// © ESI Group, 2015
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

#ifndef _WIN32
#include <pwd.h>
#endif
#include <assert.h>
#include <libpvpcap/shell.h>
#include <qchar.h>
#include <qflags.h>
#include <qlist.h>
#include <qstring.h>
#include <rapidjson/allocators.h>
#include <rapidjson/encodings.h>
#include <rapidjson/rapidjson.h>
#include <rapidjson/writer.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/constants.hpp>
#include <boost/algorithm/string/detail/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <unordered_set>
#include <QDir>
#include <QFileInfo>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <regex>

#include "../include/libpvpcap.h"
#include "../include/libpvpcap/ws.h"
#include "../include/libpvpcap/exception.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"

#include <pvlogger.h>

namespace pvpcap
{

// https://github.com/wireshark/wireshark/blob/29d48ec873e8a17280b9295159f7a946cf5ad558/wsutil/filesystem.c#L1230

std::string get_wireshark_profiles_dir()
{
	std::string wireshark_profile_dir;

	struct stat info;
	std::string homedir;
#ifndef _WIN32
	if ((homedir = std::getenv("HOME")) == std::string()) {
		homedir = getpwuid(getuid())->pw_dir;
	}
#else
	homedir = std::string(std::getenv("HOMEDRIVE")) + "/" + std::getenv("HOMEPATH");
#endif

	// Check if "$XDG_CONFIG_HOME/wireshark" exists (but don't use env var in flatpak)
	wireshark_profile_dir = homedir + "/.config/wireshark/profiles";
	if (stat(wireshark_profile_dir.c_str(), &info) == 0 and S_ISDIR(info.st_mode)) {
		return wireshark_profile_dir;
	}

	// Check if "~/.wireshark" exists for backward compatibility
	wireshark_profile_dir = homedir + "/.wireshark/profiles";
	if (stat(wireshark_profile_dir.c_str(), &info) == 0 and S_ISDIR(info.st_mode)) {
		return wireshark_profile_dir;
	}

	return wireshark_profile_dir;
}

std::vector<std::string> get_wireshark_profiles_paths()
{
	const std::string& wireshark_profiles_dir = get_wireshark_profiles_dir();
	std::vector<std::string> wireshark_profiles;
	QDir dir(wireshark_profiles_dir.c_str());
	for (const QString& profile_name :
	     dir.entryList(QDir::Dirs | QDir::Readable | QDir::NoDotAndDotDot, QDir::Name)) {
		QString profile_path(QString::fromStdString(wireshark_profiles_dir) + QDir::separator() +
		                     profile_name);
		wireshark_profiles.emplace_back(profile_path.toStdString());
	}
	return wireshark_profiles;
}

std::string ws_protocols_dict_path()
{
	return get_user_conf_dir() + "/ws_protocols_dict";
}

rapidjson::Document ws_parse_protocol_dict(const std::string& protocols_dict_file)
{
	rapidjson::Document protocol_dict;

	// Open protocols_dict json file
	std::ifstream ifs(std::filesystem::path{protocols_dict_file});

	if (not ifs) {
		throw PcapTreeException("Unable to open protocol_dict json file");
	}

	std::string content{std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>()};

	// json DOM
	protocol_dict.Parse(content.c_str());

	if (protocol_dict.HasParseError()) {
		throw PcapTreeException("protocol_dict json parse error");
	}

	return protocol_dict;
}

/*******************************************************************************
 *
 * ws_get_version
 *
 ******************************************************************************/
std::string ws_get_version()
{
	// Save tshark version. we use the first line
	std::string tshark_cmd = std::string(tshark_path()) + " -v";
	std::vector<std::string> tshark_version = execute_cmd(tshark_cmd);

	// Extract version number
    std::regex version_regex(R"(\d+\.\d+\.\d+)");
    std::smatch match;
    if (not std::regex_search(tshark_version[0], match, version_regex)) {
		pvlogger::error() << "Unable to detect tshark version" << std::endl;
    }
	return match.str().c_str();
}

/*******************************************************************************
 *
 * ws_create_protocols_dict
 *
 ******************************************************************************/
void ws_create_protocols_dict(std::string const& protocols_dict_file)
{
	rapidjson::Document json;
	json.SetObject();
	rapidjson::Document::AllocatorType& alloc = json.GetAllocator();
	std::vector<std::string> split_fields;

	rapidjson::Value version;
	std::string tshark_version = ws_get_version();
	version.SetString(tshark_version.c_str(), alloc);
	json.AddMember("tshark_version", version, alloc);

	// Add protocols information : key = protocol filter name
	std::vector<std::string> protocols = ws_get_tshark_protocols();
	for (auto& p : protocols) {
		p = p.substr(0, p.length() - 1); // delete '\n'
		boost::split(split_fields, p, boost::is_any_of("\t"));

		rapidjson::Value val;
		rapidjson::Value obj;
		obj.SetObject();

		val.SetString(split_fields[0].c_str(), alloc);
		obj.AddMember("name", val, alloc);

		val.SetString(split_fields[1].c_str(), alloc);
		obj.AddMember("short_name", val, alloc);

		val.SetString(split_fields[2].c_str(), alloc);
		obj.AddMember("filter_name", val, alloc);

		obj.AddMember("fields", rapidjson::Value(rapidjson::kArrayType), alloc);

		rapidjson::Value filter_name(split_fields[2].c_str(), split_fields[2].size(), alloc);
		json.AddMember(filter_name, obj, alloc);
	}

	// Add the fields of every protocols
	std::vector<std::string> fields = ws_get_tshark_fields();
	for (auto& f : fields) {
		f = f.substr(0, f.length() - 1); // delete '\n'
		boost::split(split_fields, f, boost::is_any_of("\t"));

		rapidjson::Value val;
		rapidjson::Value obj;
		obj.SetObject();

		// Find protocol filter_name  (split_fields[4])
		rapidjson::Value::MemberIterator it = json.FindMember(split_fields[4].c_str());
		if (it != json.MemberEnd()) {

			val.SetString(split_fields[1].c_str(), alloc);
			obj.AddMember("name", val, alloc);

			val.SetString(split_fields[2].c_str(), alloc);
			obj.AddMember("filter_name", val, alloc);

			val.SetString(split_fields[3].c_str(), alloc);
			obj.AddMember("type", val, alloc);

			val.SetString(split_fields[7].c_str(), alloc);
			obj.AddMember("description", val, alloc);

			// it's the best way to add select tag here.
			// we will use this tag to select or unselect the field object later
			val.SetBool(false);
			obj.AddMember("select", val, alloc);

			it->value["fields"].GetArray().PushBack(obj, alloc);
		}
	}

	// save json to file
	rapidjson::StringBuffer str_buffer;
	rapidjson::Writer<rapidjson::StringBuffer> writer(str_buffer);
	json.Accept(writer);

	std::ofstream ofs{std::filesystem::path(protocols_dict_file)};
	ofs << str_buffer.GetString();

	if (!ofs.good()) {
		throw PcapTreeException("Can't write the JSON string to the file!");
	}
}

/*******************************************************************************
 *
 * build_protocols_dict
 *
 ******************************************************************************/

rapidjson::Document ws_create_protocols_tree(const std::string& source,
                                             const rapidjson::Document& protocols_dict)
{
	/*
	 * We use the tshark hierarchy statistics to build the protocol tree.
	 * Here is an example of the output of the command :
	 *
	 *>>>
	 * $ tshark -q -z io,phs,frame -r demo_02.pcapng
	 *
	 * ===================================================================
	 * Protocol Hierarchy Statistics
	 * Filter: frame
	 *
	 * frame                                    frames:19246 bytes:20524313
	 *   eth                                    frames:19246 bytes:20524313
	 *     ip                                   frames:19246 bytes:20524313
	 *       tcp                                frames:19235 bytes:20522161
	 *         http                             frames:10 bytes:5989
	 *           data-text-lines                frames:1 bytes:118
	 *             tcp.segments                 frames:1 bytes:118
	 *           image-gif                      frames:2 bytes:916
	 *           xml                            frames:1 bytes:487
	 *           media                          frames:1 bytes:1261
	 *             tcp.segments                 frames:1 bytes:1261
	 *       ipv6                               frames:11 bytes:2152
	 *         tcp                              frames:11 bytes:2152
	 *           http                           frames:2 bytes:1282
	 *             image-gif                    frames:1 bytes:539
	 * ===================================================================
	 *>>>
	 *
	 * The aim is to read that output line by line, apply some transforms
	 * and add it to the protocol tree.
	 *
	 */
	rapidjson::Document document;
	rapidjson::Document::AllocatorType& alloc = document.GetAllocator();
	document.SetObject();
	document.AddMember("children", rapidjson::Value(rapidjson::kArrayType), alloc);

	/*
	 * Now we use 3 variables to help us building the tree
	 */
	std::vector<rapidjson::Value*> parents; // save current parents hierarchy
	std::vector<std::size_t> indentations;  // save the positions of the current parents hierarchy
	std::size_t position;                   // the current position
	std::size_t indent_position = 0;        // the "..." position

	// we define the root node. It's an empty item and its parent is on nullptr.
	// It's necessary for Qt QAbstractItemModel managing.
	parents.emplace_back(&document);
	indentations.push_back(0);

	// We begin by ignoring the superfluous lines
	std::string row;
	std::ifstream infile(std::filesystem::path{source});
	size_t row_idx = 0;
	while (std::getline(infile, row)) {
		if (++row_idx > 5 and row[0] != '=') { // skip 5 rows header and 1 row footer

			// find the position of the next protocol
			// We go through the whole line until we find a non-null character.
			position = row.find_first_not_of(' ');

			// We have found the position of a new protocol, we split the line
			// and escape all the space characters.
			// we get an array of 3 columns : "protocol_filter,frames,bytes"
			std::vector<std::string> split_row;
			boost::trim(row);
			boost::split(split_row, row, boost::is_any_of(" "), boost::token_compress_on);

			/*
			 * Get protocol filter
			 */
			std::string protocol_filter = split_row[0];
			// sometimes a row begin with "..."
			// it means indent
			if (protocol_filter.substr(0, 3) == "...") {
				protocol_filter = protocol_filter.substr(3, protocol_filter.length() - 3);
				indent_position += 2; // two spaces indentation
			} else {
				indent_position = 0;
			}
			position += indent_position;

			/*
			 * Buid protocol object with previous information
			 */
			rapidjson::Value val;
			rapidjson::Value protocol;
			protocol.SetObject();

			rapidjson::Value::ConstMemberIterator it =
			    protocols_dict.FindMember(protocol_filter.c_str());
			if (it != protocols_dict.MemberEnd()) {
				assert(it->value.IsObject() && "The protocol found is not an json object");

				/*
				 * Get frames : number of packets
				 */
				std::string prefix_packets = "frames:";
				std::string packets_str = split_row[1];
				std::size_t packets = std::stoll(packets_str.erase(0, prefix_packets.length()));

				/*
				 * Get bytes : size of packets
				 */
				std::string prefix_bytes = "bytes:";
				std::string bytes_str = split_row[2];
				std::size_t bytes = std::stoll(bytes_str.erase(0, prefix_bytes.length()));

				val.SetString(protocol_filter.c_str(), alloc);
				protocol.AddMember("filter_name", val, alloc);

				val.SetUint64(bytes);
				protocol.AddMember("bytes", val, alloc);

				val.SetUint64(packets);
				protocol.AddMember("packets", val, alloc);

				protocol.AddMember("children", rapidjson::Value(rapidjson::kArrayType), alloc);

				if (protocol_filter == "frame") {
					// frame has the total bytes and total packets. We save it to root node
					val.SetUint64(bytes);
					document.AddMember("bytes", val, alloc);
					val.SetUint64(packets);
					document.AddMember("packets", val, alloc);
				}
			} else {
				continue; // sometime tshark output fields as they were protocols...
			}

			if (position > indentations.back()) {
				// The last child of the current parent is now the new parent
				// unless the current parent has no children
				rapidjson::Value& children = (*parents.back())["children"];
				if (not children.Empty()) {
					parents.emplace_back(&children[children.Size() - 1]);
					indentations.push_back(position);
				}
			} else {
				while (position < indentations.back() && parents.size() > 0) {
					parents.pop_back();
					indentations.pop_back();
				}
			}

			// Append a new item to the current parent's list of children.
			rapidjson::Value& children = (*parents.back())["children"];
			children.GetArray().PushBack(protocol, alloc);
		}
	}

	return document;
}

static void ws_merge_protocols_tree_rec(rapidjson::Value& dst,
                                        const rapidjson::Value& src,
                                        rapidjson::Document::AllocatorType& allocator)
{
	const rapidjson::Value& src_children = src["children"];
	rapidjson::Value& dst_children = dst["children"];

	for (auto src_array_it = src_children.Begin(); src_array_it != src_children.End();
	     ++src_array_it) {

		const std::string& src_protocol = (*src_array_it)["filter_name"].GetString();

		bool protocol_found = false;
		for (auto dst_array_it = dst_children.Begin(); dst_array_it != dst_children.End();
		     ++dst_array_it) {

			if ((*dst_array_it)["filter_name"].GetString() == src_protocol) {
				(*dst_array_it)["bytes"].SetUint64((*dst_array_it)["bytes"].GetUint64() +
				                                   (*src_array_it)["bytes"].GetUint64());
				(*dst_array_it)["packets"].SetUint64((*dst_array_it)["packets"].GetUint64() +
				                                     (*src_array_it)["packets"].GetUint64());

				if (not(*dst_array_it)["children"].Empty()) {
					ws_merge_protocols_tree_rec((*dst_array_it), (*src_array_it), allocator);
				}
				protocol_found = true;
				break;
			}
		}

		if (not protocol_found) {
			rapidjson::Value protocol;
			protocol.CopyFrom((*src_array_it), allocator);
			dst_children.PushBack(protocol, allocator);
		}
	}
}

void ws_merge_protocols_tree(rapidjson::Value& dst,
                             const rapidjson::Value& src,
                             rapidjson::Document::AllocatorType& allocator)
{
	dst["bytes"].SetUint64(dst["bytes"].GetUint64() + src["bytes"].GetUint64());
	dst["packets"].SetUint64(dst["packets"].GetUint64() + src["packets"].GetUint64());

	ws_merge_protocols_tree_rec(dst, src, allocator);
}

static void ws_enrich_protocols_tree_rec(rapidjson::Value& enriched_protocols_tree,
                                         const rapidjson::Document& protocols_dict,
                                         rapidjson::Document::AllocatorType& alloc)
{
	rapidjson::Value& children = enriched_protocols_tree["children"];

	for (auto it = children.Begin(); it != children.End(); ++it) {

		const std::string& filter_name = (*it)["filter_name"].GetString();

		auto proto_it = protocols_dict.FindMember(filter_name.c_str());

		rapidjson::Value val;

		val.SetString(proto_it->value["name"].GetString(), alloc);
		(*it).AddMember("name", val, alloc);

		val.SetString(proto_it->value["short_name"].GetString(), alloc);
		(*it).AddMember("short_name", val, alloc);

		val.CopyFrom(proto_it->value["fields"], alloc);
		(*it).AddMember("fields", val, alloc);

		ws_enrich_protocols_tree_rec((*it), protocols_dict, alloc);
	}
}

void ws_enrich_protocols_tree(rapidjson::Document& enriched_protocols_tree,
                              const rapidjson::Document& protocols_tree,
                              const rapidjson::Document& protocols_dict)
{
	rapidjson::Document::AllocatorType& alloc = enriched_protocols_tree.GetAllocator();

	enriched_protocols_tree.AddMember("bytes", protocols_tree["bytes"].GetUint64(), alloc);
	enriched_protocols_tree.AddMember("packets", protocols_tree["packets"].GetUint64(), alloc);
	enriched_protocols_tree["children"].CopyFrom(protocols_tree["children"], alloc);

	ws_enrich_protocols_tree_rec(enriched_protocols_tree, protocols_dict, alloc);
}

/*******************************************************************************
 *
 * ws_get_tshark_fields
 *
 ******************************************************************************/
std::vector<std::string> ws_get_tshark_fields()
{
	std::string cmdline = tshark_path() + " -G fields";
	std::vector<std::string> fields = execute_cmd(cmdline);

	// delete all line who doesn't begin with "F"
	auto i = fields.begin();
	while (i != fields.end()) {
		if ((*i).at(0) != 'F')
			i = fields.erase(i); // delete all not fields line
		else
			++i;
	}

	return fields;
}

/*******************************************************************************
 *
 * ws_get_tshark_protocols
 *
 ******************************************************************************/
std::vector<std::string> ws_get_tshark_protocols()
{
	std::string cmdline = tshark_path() + " -G protocols";
	return execute_cmd(cmdline);
}

/*******************************************************************************
 *
 * ws_get_selected_field_list
 *
 ******************************************************************************/
static void visit_fields(const rapidjson::Value& children,
                         const std::function<void(const rapidjson::Value&)>& f,
                         bool all)
{
	rapidjson::Document::AllocatorType alloc;

	for (auto& child : children.GetArray()) {

		for (auto& field : child["fields"].GetArray()) {
			if (all or field["select"].GetBool()) {
				f(field);
			}
		}

		// recurse over children
		visit_fields(child["children"], f, all);
	}
}

static void visit_fields(const rapidjson::Document& json_data,
                         const std::function<void(const rapidjson::Value& json_data)>& f,
                         bool all = false)
{
	visit_fields(json_data["children"], f, all);
}

rapidjson::Document ws_get_selected_fields(const rapidjson::Document& json_data)
{
	rapidjson::Document selected_fields;
	selected_fields.SetObject();

	rapidjson::Document::AllocatorType& alloc = selected_fields.GetAllocator();

	visit_fields(json_data, [&](const rapidjson::Value& field) {

		rapidjson::Value::MemberIterator it =
		    selected_fields.FindMember(field["filter_name"].GetString());
		if (it == selected_fields.MemberEnd()) {
			rapidjson::Value field_value(field, alloc);
			rapidjson::Value filter_name(field["filter_name"].GetString(),
			                             strlen(field["filter_name"].GetString()), alloc);
			selected_fields.AddMember(filter_name, field_value, alloc);
		}
	});

	return selected_fields;
}

std::vector<std::string> ws_get_cmdline_opts(rapidjson::Document& json_data)
{
	std::vector<std::string> ts_fields;

	// Removed duplicated fields
	std::unordered_set<std::string> distinct_fields;
	visit_fields(json_data, [&](const rapidjson::Value& field) {
		const std::string& field_name = field["filter_name"].GetString();
		if (distinct_fields.find(field_name) == distinct_fields.end()) {
			ts_fields.emplace_back(field_name);
			distinct_fields.emplace(field_name);
		}
	});

	// add wireshark specials fields
	if (json_data["options"]["source"].GetBool())
		ts_fields.emplace_back(ws_map_special_fields.at("source"));

	if (json_data["options"]["destination"].GetBool())
		ts_fields.emplace_back(ws_map_special_fields.at("destination"));

	if (json_data["options"]["protocol"].GetBool())
		ts_fields.emplace_back(ws_map_special_fields.at("protocol"));

	if (json_data["options"]["info"].GetBool())
		ts_fields.emplace_back(ws_map_special_fields.at("info"));

	// get rewrite options
	const std::string& header = (json_data["options"]["header"].GetBool() ? "y" : "n");
	const std::string& occurrence = json_data["options"]["occurrence"].GetString();
	const std::string& aggregator = json_data["options"]["aggregator"].GetString();
	const std::string& filters = json_data["options"]["filters"].GetString();

	std::vector<std::string> opts;
	opts.emplace_back(tshark_path());
	if (not filters.empty()) {
		opts.emplace_back("-Y" + filters);
	}
	opts.emplace_back("-Tfields");
	for (const std::string& field : ts_fields) {
		opts.emplace_back("-e" + field);
	}
	opts.emplace_back("-Eheader=" + header);
	opts.emplace_back(std::string("-Eseparator=") + pvpcap::SEPARATOR);
	opts.emplace_back("-Eoccurrence=" + occurrence);
	opts.emplace_back("-Eaggregator=" + aggregator);
	opts.emplace_back(std::string("-Equote=") + pvpcap::QUOTE);

	// TCP/IP
	if (json_data["options"].HasMember("tcp.desegment_tcp_streams") &&
	    not json_data["options"]["tcp.desegment_tcp_streams"].GetBool()) {
		opts.emplace_back("-otcp.desegment_tcp_streams:FALSE");
	}
	if (json_data["options"].HasMember("ip.defragment") &&
	    not json_data["options"]["ip.defragment"].GetBool()) {
		opts.emplace_back("-oip.defragment:FALSE");
	}

	// Name resolution (see https://www.wireshark.org/docs/man-pages/tshark.html)
	std::string name_resolving_flags;
	if (json_data["options"].HasMember("nameres.network_name") &&
	    json_data["options"]["nameres.network_name"].GetBool()) {
		name_resolving_flags += "n";
	}
	if (json_data["options"].HasMember("nameres.dns_pkt_addr_resolution") &&
	    json_data["options"]["nameres.dns_pkt_addr_resolution"].GetBool()) {
		name_resolving_flags += "d";
	}
	if (json_data["options"].HasMember("nameres.use_external_name_resolver") &&
	    json_data["options"]["nameres.use_external_name_resolver"].GetBool()) {
		name_resolving_flags += "N";
	}
	std::string geoip_db_paths = "/var/run/host/usr/share/GeoIP/";
	if (json_data["options"].HasMember("geoip_db_paths")) {
		geoip_db_paths = json_data["options"]["geoip_db_paths"].GetString();
	}
	QString geoip_db_paths_filename = "~/.wireshark/maxmind_db_paths";
	geoip_db_paths_filename.replace(QString('~'), QDir::homePath());
	QDir().mkdir(QFileInfo(geoip_db_paths_filename).dir().path());
	std::ofstream geoip_db_paths_file(std::filesystem::path(geoip_db_paths_filename.toUtf8().constData()), std::ios_base::trunc);
	geoip_db_paths_file << "\"" << geoip_db_paths << "\"" << std::endl;
	if (not name_resolving_flags.empty()) {
		opts.emplace_back("-N" + name_resolving_flags);
	}

	// disable some tcp reconstructions options
	opts.emplace_back("-otcp.summary_in_tree:TRUE");
	opts.emplace_back("-otcp.check_checksum:FALSE");
	opts.emplace_back("-otcp.analyze_sequence_numbers:TRUE");
	opts.emplace_back("-otcp.relative_sequence_numbers:FALSE");
	opts.emplace_back("-otcp.track_bytes_in_flight:FALSE");
	opts.emplace_back("-otcp.calculate_timestamps:FALSE");

	// Add wireshark profile if any
	if (json_data["options"].HasMember("wireshark_profile")) {
		const std::string& wireshark_profile =
		    json_data["options"]["wireshark_profile"].GetString();
		opts.emplace_back("-C" + wireshark_profile);
	}

	opts.emplace_back("-r-");

	return opts;
}

} /* namespace pvpcap */
