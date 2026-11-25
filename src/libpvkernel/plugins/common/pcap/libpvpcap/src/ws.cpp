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
#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <array>
#include <cctype>
#include <string_view>
#include <regex>

#include "../include/libpvpcap.h"
#include "../include/libpvpcap/capture.h"
#include "../include/libpvpcap/process.h"
#include "../include/libpvpcap/ws.h"
#include "../include/libpvpcap/exception.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"

#include <pvlogger.h>

namespace pvpcap
{

// https://github.com/wireshark/wireshark/blob/29d48ec873e8a17280b9295159f7a946cf5ad558/wsutil/filesystem.c#L1230

/* getenv() returns nullptr for a variable that is not set, and building a
 * std::string out of it throws std::logic_error ("basic_string: construction
 * from null is not valid"), which used to abort the whole process. Windows CI
 * runners are a real case of this: they leave HOMEDRIVE and HOMEPATH unset.
 */
static std::string getenv_or_empty(const char* name)
{
	const char* value = std::getenv(name);
	return value != nullptr ? std::string(value) : std::string();
}

std::string get_wireshark_profiles_dir()
{
	std::string wireshark_profile_dir;

	struct stat info;
	std::string homedir;
#ifndef _WIN32
	homedir = getenv_or_empty("HOME");
	if (homedir.empty()) {
		// getpwuid() itself returns nullptr when the uid has no passwd entry
		const struct passwd* pw = getpwuid(getuid());
		if (pw != nullptr and pw->pw_dir != nullptr) {
			homedir = pw->pw_dir;
		}
	}
#else
	// HOMEDRIVE ("C:") and HOMEPATH ("\Users\someone") are meant to be concatenated
	homedir = getenv_or_empty("HOMEDRIVE") + getenv_or_empty("HOMEPATH");
	if (homedir.empty()) {
		homedir = getenv_or_empty("USERPROFILE");
	}
#endif

	if (homedir.empty()) {
		// no home directory to look into: no wireshark profile can be found
		return wireshark_profile_dir;
	}

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

/*******************************************************************************
 *
 * ws_get_version
 *
 ******************************************************************************/
bool wireshark_profile_exists(const std::string& name)
{
	return name.empty() or
	       QFileInfo(QString::fromStdString(get_wireshark_profiles_dir() + "/" + name)).isDir();
}

std::string ws_get_version()
{
	// The version number is on the first line.
	const std::vector<std::string> lines = tshark_output({"-v"});
	std::regex version_regex(R"(\d+\.\d+\.\d+)");
	std::smatch match;
	if (lines.empty() or not std::regex_search(lines.front(), match, version_regex)) {
		pvlogger::error() << "Unable to detect tshark version" << std::endl;
		return {};
	}
	return match.str();
}

std::vector<std::string> tshark_output(const std::vector<std::string>& arguments)
{
	QTemporaryDir directory;
	if (not directory.isValid()) {
		return {};
	}
	child_process::options process;
	process.argv.push_back(tshark_path());
	process.argv.insert(process.argv.end(), arguments.begin(), arguments.end());
	process.output = directory.filePath("output").toStdString();
	process.errors = directory.filePath("errors").toStdString();
	process.environment = tshark_environment();

	try {
		child_process tshark(process);
		tshark.close_input();
		const child_process::status status = tshark.wait();
		if (not status.exited or status.code != 0) {
			return {};
		}
	} catch (const process_error& e) {
		pvlogger::error() << e.what() << std::endl;
		return {};
	}

	std::vector<std::string> lines;
	std::ifstream output{std::filesystem::path(directory.filePath("output").toStdU16String())};
	for (std::string line; std::getline(output, line);) {
		if (not line.empty() and line.back() == '\r') {
			line.pop_back();
		}
		lines.push_back(std::move(line));
	}
	return lines;
}

filter_check ws_check_filter(const std::string& filter)
{
	filter_check check;
	QTemporaryDir directory;
	if (not directory.isValid()) {
		return check;
	}
	// A capture of no frame: tshark only checks the filter.
	const std::string capture = directory.filePath("empty.pcap").toStdString();
	{
		QFile file(QString::fromStdString(capture));
		const uint32_t header[] = {0xa1b2c3d4, 0x00040002, 0, 0, 65535, 1};
		if (not file.open(QIODevice::WriteOnly) or
		    file.write(reinterpret_cast<const char*>(header), sizeof(header)) != sizeof(header)) {
			return check;
		}
	}
	child_process::options process;
	process.argv = {tshark_path(), "-r", capture, "-Y", filter};
	process.output = directory.filePath("output").toStdString();
	process.errors = directory.filePath("errors").toStdString();
	process.environment = tshark_environment();
	child_process::status status;
	try {
		child_process tshark(process);
		tshark.close_input();
		status = tshark.wait();
	} catch (const process_error&) {
		return check;
	}
	check.checked = true;
	if (status.exited and status.code == 0) {
		return check;
	}

	/* Such as:
	 * tshark: "http.reqest" is not a valid protocol or protocol field.
	 *     http.reqest
	 *     ^~~~~~~~~~~
	 */
	const std::string errors = tshark_errors(directory.filePath("errors").toStdString());
	std::vector<std::string> lines;
	boost::split(lines, errors, boost::is_any_of("\n"));
	for (std::string& line : lines) {
		if (not line.empty() and line.back() == '\r') {
			line.pop_back();
		}
	}
	if (lines.front().empty()) {
		check.problem = "tshark can't read this filter.";
		return check;
	}
	check.problem = lines.front().starts_with("tshark: ") ? lines.front().substr(8) : lines.front();
	for (size_t i = 1; i < lines.size(); i++) {
		const size_t caret = lines[i].find('^');
		if (caret != std::string::npos and lines[i].find_first_not_of(' ') == caret and
		    caret >= 4) {
			check.position = caret - 4;
			check.length = lines[i].find_first_not_of("^~", caret);
			check.length =
			    (check.length == std::string::npos ? lines[i].size() : check.length) - caret;
			break;
		}
	}
	return check;
}

std::unordered_map<std::string, field_description>
ws_field_descriptions(const std::vector<std::string>& names)
{
	const std::unordered_set<std::string> wanted(names.begin(), names.end());
	std::unordered_map<std::string, field_description> descriptions;
	std::vector<std::string> columns;
	for (const std::string& line : tshark_output({"-G", "fields"})) {
		if (not line.starts_with("F\t")) {
			continue;
		}
		boost::split(columns, line, boost::is_any_of("\t"));
		if (columns.size() > 5 and wanted.contains(columns[2])) {
			descriptions[columns[2]] = {columns[3], columns[5]};
		}
	}
	return descriptions;
}

/*******************************************************************************
 *
 * ws_get_tshark_fields
 *
 ******************************************************************************/
std::vector<std::string> ws_get_tshark_fields()
{
	std::vector<std::string> fields;
	for (std::string& line : tshark_output({"-G", "fields"})) {
		if (line.starts_with("F\t")) {
			fields.push_back(std::move(line));
		}
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
	return tshark_output({"-G", "protocols"});
}

namespace
{

//! Numbers of earlier frames, which a single pass writes as two do.
constexpr std::array<std::string_view, 7> ONE_PASS_FRAMES = {"dns.response_to",
                                                             "http.request_in",
                                                             "icmp.resp_to",
                                                             "ldap.response_to",
                                                             "sip.response-request",
                                                             "tcp.analysis.acks_frame",
                                                             "tcp.analysis.duplicate_ack_frame"};

constexpr std::array<std::string_view, 9> TWO_PASS_FIELDS = {
    "_ws.col.Info",          "_ws.col.info",        "dns.response_missing",
    "icmp.no_resp",          "icmp.resp_not_found", "icmpv6.no_resp",
    "icmpv6.resp_not_found", "ntp.delta_time",      "tcp.analysis.initial_rtt"};

constexpr std::array<std::string_view, 6> TWO_PASS_PREFIXES = {
    "_ws.expert", "dcerpc.", "smb.", "smb2.", "tcp.completeness", "tcp.stream."};

extraction::field::kind field_kind(const std::string& name, const std::string& type)
{
	if (name == "frame.time_relative") {
		return extraction::field::kind::time_since_first;
	}
	if (name == "frame.time_delta") {
		return extraction::field::kind::time_since_previous;
	}
	if (name == "frame.time_delta_displayed") {
		return extraction::field::kind::time_since_shown;
	}
	if (canonical_field_name(name) == "_ws.col.Info") {
		return extraction::field::kind::summary;
	}
	if (type == "FT_FRAMENUM") {
		return extraction::field::kind::frames;
	}
	if (type == "FT_ABSOLUTE_TIME") {
		return extraction::field::kind::absolute_time;
	}
	return extraction::field::kind::text;
}

} // namespace

bool ws_needs_two_passes(const std::string& name, const std::string& type)
{
	if (type == "FT_FRAMENUM") {
		return std::find(ONE_PASS_FRAMES.begin(), ONE_PASS_FRAMES.end(), name) ==
		       ONE_PASS_FRAMES.end();
	}
	return std::find(TWO_PASS_FIELDS.begin(), TWO_PASS_FIELDS.end(), name) !=
	           TWO_PASS_FIELDS.end() or
	       std::any_of(TWO_PASS_PREFIXES.begin(), TWO_PASS_PREFIXES.end(),
	                   [&](std::string_view prefix) { return name.starts_with(prefix); });
}

std::vector<std::string> ws_filter_fields(const std::string& filter)
{
	const auto in_name = [](char c) {
		return std::isalnum(static_cast<unsigned char>(c)) or c == '_' or c == '.' or c == '-';
	};
	std::vector<std::string> names;
	size_t i = 0;
	while (i < filter.size()) {
		const char c = filter[i];
		if (c == '"' or c == '\'') {
			for (i++; i < filter.size() and filter[i] != c; i++) {
				i += filter[i] == '\\';
			}
			i++;
		} else if (in_name(c)) {
			const size_t begin = i;
			while (i < filter.size() and in_name(filter[i])) {
				i++;
			}
			names.push_back(filter.substr(begin, i - begin));
		} else {
			i++;
		}
	}
	return names;
}

std::vector<std::pair<std::string, std::string>> tshark_environment()
{
	// Times tshark writes as local come in UTC, which is how the formats read them.
	std::vector<std::pair<std::string, std::string>> environment = {{"TZ", "UTC0"}};
#ifdef __linux__
	// Within Flatpak, XDG_CONFIG_HOME points inside the sandbox: tshark has to be told where
	// Wireshark keeps the configuration of the user, its profiles included.
	const std::string home = getenv_or_empty("HOME");
	if (qEnvironmentVariableIsSet("FLATPAK_ID") and not home.empty() and
	    QFileInfo(QString::fromStdString(home + "/.config/wireshark")).isDir()) {
		environment.emplace_back("XDG_CONFIG_HOME", home + "/.config");
	}
#endif
	return environment;
}

std::vector<std::string> ws_reading_options(const profile& p)
{
	bool tcp_times = false;
	bool bytes_in_flight = false;
	for (const profile_field& f : p.columns) {
		tcp_times |= f.name.starts_with("tcp.time_");
		bytes_in_flight |=
		    f.name == "tcp.analysis.bytes_in_flight" or f.name == "tcp.analysis.push_bytes_sent";
	}

	std::vector<std::string> args;
	if (not p.reassemble_tcp) {
		args.insert(args.end(), {"-o", "tcp.desegment_tcp_streams:FALSE"});
	}
	if (not p.defragment_ip) {
		args.insert(args.end(), {"-o", "ip.defragment:FALSE"});
	}

	// See https://www.wireshark.org/docs/man-pages/tshark.html
	std::string name_resolution;
	if (p.resolve_names) {
		name_resolution += "n";
	}
	if (p.names_from_captures) {
		name_resolution += "d";
	}
	if (p.names_from_servers) {
		name_resolution += "N";
	}
	if (not name_resolution.empty()) {
		args.insert(args.end(), {"-N", name_resolution});
	}

	// Through an option, rather than in a file of the configuration of the user, which
	// Wireshark reads too.
	const std::string geoip =
	    p.geoip_directory.empty() ? DEFAULT_GEOIP_DIRECTORY : p.geoip_directory;
	if (QFileInfo(QString::fromStdString(geoip)).isDir()) {
		args.insert(args.end(), {"-o", "uat:maxmind_db_paths:\"" + geoip + "\""});
	}

	args.insert(args.end(),
	            {"-o", "tcp.summary_in_tree:TRUE", "-o", "tcp.check_checksum:FALSE", "-o",
	             "tcp.analyze_sequence_numbers:TRUE", "-o", "tcp.relative_sequence_numbers:FALSE",
	             "-o",
	             std::string("tcp.track_bytes_in_flight:") + (bytes_in_flight ? "TRUE" : "FALSE"),
	             "-o", std::string("tcp.calculate_timestamps:") + (tcp_times ? "TRUE" : "FALSE")});

	// tshark would refuse to read the captures at all.
	if (not p.wireshark_profile.empty() and wireshark_profile_exists(p.wireshark_profile)) {
		args.insert(args.end(), {"-C", p.wireshark_profile});
	}
	return args;
}

extraction::options ws_extraction_options(const profile& p)
{
	extraction::options opts;
	for (const profile_field& f : p.columns) {
		opts.fields.push_back({f.name, field_kind(f.name, f.type)});
		opts.two_pass |= ws_needs_two_passes(f.name, f.type);
	}

	std::vector<std::string>& args = opts.tshark_options;
	if (not p.filter.empty()) {
		args.insert(args.end(), {"-Y", p.filter});
		// A single pass would keep other frames than such a filter says.
		if (not opts.two_pass) {
			const std::vector<std::string> names = ws_filter_fields(p.filter);
			const auto descriptions = ws_field_descriptions(names);
			for (const std::string& name : names) {
				const auto it = descriptions.find(name);
				opts.two_pass |=
				    ws_needs_two_passes(name, it != descriptions.end() ? it->second.type : "");
			}
		}
	}
	opts.aggregator = p.aggregator;
	args.insert(args.end(), {"-E", std::string("occurrence=") + p.occurrence, "-E",
	                         "aggregator=" + opts.aggregator});
	const std::vector<std::string> reading = ws_reading_options(p);
	args.insert(args.end(), reading.begin(), reading.end());

	opts.environment = tshark_environment();
	return opts;
}

} /* namespace pvpcap */
