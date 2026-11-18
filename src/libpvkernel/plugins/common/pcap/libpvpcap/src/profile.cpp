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

#include "../include/libpvpcap/profile.h"

#include "../include/libpvpcap/shell.h"
#include "../include/libpvpcap/ws.h"

#include <QFile>
#include <QRegularExpression>
#include <QSaveFile>
#include <QString>

#include <rapidjson/error/en.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

#include <unordered_map>
#include <unordered_set>

namespace pvpcap
{

namespace
{

constexpr int FORMAT_VERSION = 2;

const rapidjson::Value* member(const rapidjson::Value& object, const char* name)
{
	if (not object.IsObject()) {
		return nullptr;
	}
	const auto it = object.FindMember(name);
	return it != object.MemberEnd() ? &it->value : nullptr;
}

bool option(const rapidjson::Value& object, const char* name, bool fallback)
{
	const rapidjson::Value* value = member(object, name);
	return value != nullptr and value->IsBool() ? value->GetBool() : fallback;
}

std::string option(const rapidjson::Value& object, const char* name, const char* fallback)
{
	const rapidjson::Value* value = member(object, name);
	return value != nullptr and value->IsString()
	           ? std::string(value->GetString(), value->GetStringLength())
	           : fallback;
}

//! The fields earlier versions selected in their tree of protocols, in the order of the tree.
void selected_fields(const rapidjson::Value* protocols,
                     std::vector<profile_field>& fields,
                     size_t depth = 0)
{
	if (protocols == nullptr or not protocols->IsArray() or depth > 256) {
		return;
	}
	for (const rapidjson::Value& protocol : protocols->GetArray()) {
		const rapidjson::Value* list = member(protocol, "fields");
		if (list != nullptr and list->IsArray()) {
			for (const rapidjson::Value& field : list->GetArray()) {
				if (option(field, "select", false)) {
					fields.push_back({option(field, "filter_name", ""), option(field, "type", "")});
				}
			}
		}
		selected_fields(member(protocol, "children"), fields, depth + 1);
	}
}

std::string json_text(const rapidjson::Value& json)
{
	rapidjson::StringBuffer buffer;
	rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
	writer.SetIndent(' ', 2);
	json.Accept(writer);
	return {buffer.GetString(), buffer.GetSize()};
}

void write_file(const std::string& text, const std::string& path)
{
	QSaveFile file(QString::fromStdString(path));
	if (not file.open(QIODevice::WriteOnly) or
	    file.write(text.data(), qint64(text.size())) != qint64(text.size()) or not file.commit()) {
		throw profile_error(file.errorString().toStdString());
	}
}

rapidjson::Document parse_file(const std::string& path)
{
	QFile file(QString::fromStdString(path));
	if (not file.open(QIODevice::ReadOnly)) {
		throw profile_error(file.errorString().toStdString());
	}
	const QByteArray content = file.readAll();
	rapidjson::Document json;
	json.Parse(content.constData(), content.size());
	if (json.HasParseError()) {
		throw profile_error(std::string("It is not JSON: ") +
		                    rapidjson::GetParseError_En(json.GetParseError()) + " (byte " +
		                    std::to_string(json.GetErrorOffset()) + ").");
	}
	return json;
}

std::string drafts_path()
{
	return get_user_conf_dir() + "/profile_drafts.json";
}

} // namespace

profile profile_from_json(const rapidjson::Value& json)
{
	if (not json.IsObject()) {
		throw profile_error("It holds no profile.");
	}
	static const rapidjson::Value no_options(rapidjson::kObjectType);
	const rapidjson::Value* options_member = member(json, "options");
	const rapidjson::Value& options =
	    options_member != nullptr and options_member->IsObject() ? *options_member : no_options;

	std::vector<profile_field> fields;
	const rapidjson::Value* columns = member(json, "columns");
	if (columns != nullptr and columns->IsArray()) {
		for (const rapidjson::Value& column : columns->GetArray()) {
			fields.push_back({option(column, "name", ""), option(column, "type", "")});
		}
	} else {
		selected_fields(member(json, "children"), fields);
		for (const char* column : {"source", "destination", "protocol", "info"}) {
			if (option(options, column, false)) {
				fields.push_back({ws_map_special_fields.at(column), "FT_STRING"});
			}
		}
	}

	profile p;
	std::unordered_set<std::string> seen;
	for (profile_field& field : fields) {
		field.name = canonical_field_name(field.name);
		if (not field.name.empty() and seen.insert(field.name).second) {
			p.columns.push_back(std::move(field));
		}
	}

	p.filter = option(options, "filters", "");
	const std::string occurrence = option(options, "occurrence", "f");
	p.occurrence = occurrence == "l" or occurrence == "a" ? occurrence.front() : 'f';
	p.aggregator = option(options, "aggregator", "|");
	p.reassemble_tcp = option(options, "tcp.desegment_tcp_streams", true);
	p.defragment_ip = option(options, "ip.defragment", true);
	p.resolve_names = option(options, "nameres.network_name", false);
	p.names_from_captures = option(options, "nameres.dns_pkt_addr_resolution", false);
	p.names_from_servers = option(options, "nameres.use_external_name_resolver", false);
	p.geoip_directory = option(options, "geoip_db_paths", "");
	p.wireshark_profile = option(options, "wireshark_profile", "");
	return p;
}

rapidjson::Document profile_to_json(const profile& p)
{
	rapidjson::Document json(rapidjson::kObjectType);
	rapidjson::Document::AllocatorType& alloc = json.GetAllocator();
	const auto text = [&](const std::string& s) {
		return rapidjson::Value(s.c_str(), rapidjson::SizeType(s.size()), alloc);
	};

	json.AddMember("version", FORMAT_VERSION, alloc);
	rapidjson::Value columns(rapidjson::kArrayType);
	for (const profile_field& field : p.columns) {
		rapidjson::Value column(rapidjson::kObjectType);
		column.AddMember("name", text(field.name), alloc);
		column.AddMember("type", text(field.type), alloc);
		columns.PushBack(column, alloc);
	}
	json.AddMember("columns", columns, alloc);

	rapidjson::Value options(rapidjson::kObjectType);
	options.AddMember("filters", text(p.filter), alloc);
	options.AddMember("occurrence", text(std::string(1, p.occurrence)), alloc);
	options.AddMember("aggregator", text(p.aggregator), alloc);
	options.AddMember("tcp.desegment_tcp_streams", p.reassemble_tcp, alloc);
	options.AddMember("ip.defragment", p.defragment_ip, alloc);
	options.AddMember("nameres.network_name", p.resolve_names, alloc);
	options.AddMember("nameres.dns_pkt_addr_resolution", p.names_from_captures, alloc);
	options.AddMember("nameres.use_external_name_resolver", p.names_from_servers, alloc);
	if (not p.geoip_directory.empty()) {
		options.AddMember("geoip_db_paths", text(p.geoip_directory), alloc);
	}
	if (not p.wireshark_profile.empty()) {
		options.AddMember("wireshark_profile", text(p.wireshark_profile), alloc);
	}
	// Earlier versions read these without checking they are there.
	for (const char* legacy : {"source", "destination", "protocol", "info", "header"}) {
		options.AddMember(rapidjson::StringRef(legacy), false, alloc);
	}
	json.AddMember("options", options, alloc);
	json.AddMember("read_only", false, alloc);
	json.AddMember("children", rapidjson::Value(rapidjson::kArrayType), alloc);
	return json;
}

std::string canonical_field_name(const std::string& name)
{
	static const std::unordered_map<std::string, std::string> aliases = {
	    {"_ws.col.def_src", "_ws.col.Source"},
	    {"_ws.col.def_dst", "_ws.col.Destination"},
	    {"_ws.col.protocol", "_ws.col.Protocol"},
	    {"_ws.col.info", "_ws.col.Info"}};
	const auto it = aliases.find(name);
	return it != aliases.end() ? it->second : name;
}

std::string profile_location::path() const
{
	return (builtin ? get_system_profile_dir() : get_user_profile_dir()) + "/" + name;
}

std::vector<profile_location> list_profiles()
{
	std::vector<profile_location> profiles;
	for (std::string& name : get_directory_files(get_system_profile_dir())) {
		profiles.push_back({true, std::move(name)});
	}
	for (std::string& name : get_directory_files(get_user_profile_dir())) {
		profiles.push_back({false, std::move(name)});
	}
	return profiles;
}

profile read_profile(const std::string& path)
{
	return profile_from_json(parse_file(path));
}

void write_profile(const profile& p, const std::string& path)
{
	write_file(json_text(profile_to_json(p)), path);
}

std::string profile_name_problem(const std::string& name)
{
	const QString text = QString::fromStdString(name);
	if (text.trimmed().isEmpty()) {
		return "A profile needs a name.";
	}
	if (text != text.trimmed()) {
		return "A profile name can't start or end with a space.";
	}
	if (text.startsWith('.') or text.endsWith('.')) {
		return "A profile name can't start or end with a dot.";
	}
	static const QString forbidden = R"(/\:*?"<>|)";
	for (const QChar c : text) {
		if (c.unicode() < 0x20 or forbidden.contains(c)) {
			return R"(A profile name can't hold any of / \ : * ? " < > |)";
		}
	}
	if (text.size() > 100) {
		return "A profile name can't be longer than 100 characters.";
	}
	// The name of a file of the profile.
	static const QRegularExpression reserved(R"(^(con|prn|aux|nul|com\d|lpt\d)(\..*)?$)",
	                                         QRegularExpression::CaseInsensitiveOption);
	if (reserved.match(text).hasMatch()) {
		return "Windows keeps this name for itself.";
	}
	return {};
}

bool user_profile_exists(const std::string& name)
{
	// Some file systems don't tell the case of letters apart.
	const QString wanted = QString::fromStdString(name);
	for (const std::string& existing : get_directory_files(get_user_profile_dir())) {
		if (QString::fromStdString(existing).compare(wanted, Qt::CaseInsensitive) == 0) {
			return true;
		}
	}
	return false;
}

std::string free_profile_name(const std::string& wanted)
{
	std::string name = wanted;
	for (int i = 2; user_profile_exists(name); i++) {
		name = wanted + " " + std::to_string(i);
	}
	return name;
}

profile_drafts read_drafts()
{
	profile_drafts drafts;
	if (not QFile::exists(QString::fromStdString(drafts_path()))) {
		return drafts;
	}
	try {
		const rapidjson::Document json = parse_file(drafts_path());
		if (not json.IsArray()) {
			return drafts;
		}
		for (const rapidjson::Value& draft : json.GetArray()) {
			const rapidjson::Value* p = member(draft, "profile");
			if (p == nullptr) {
				continue;
			}
			drafts.emplace_back(
			    profile_location{option(draft, "builtin", false), option(draft, "name", "")},
			    profile_from_json(*p));
		}
	} catch (const profile_error&) {
		// Drafts are a convenience: nothing that was saved is lost.
	}
	return drafts;
}

void write_drafts(const profile_drafts& drafts)
{
	if (drafts.empty()) {
		QFile::remove(QString::fromStdString(drafts_path()));
		return;
	}
	rapidjson::Document json(rapidjson::kArrayType);
	rapidjson::Document::AllocatorType& alloc = json.GetAllocator();
	for (const auto& [location, p] : drafts) {
		rapidjson::Value draft(rapidjson::kObjectType);
		draft.AddMember("builtin", location.builtin, alloc);
		draft.AddMember("name",
		                rapidjson::Value(location.name.c_str(),
		                                 rapidjson::SizeType(location.name.size()), alloc),
		                alloc);
		const rapidjson::Document profile_json = profile_to_json(p);
		draft.AddMember("profile", rapidjson::Value(profile_json, alloc), alloc);
		json.PushBack(draft, alloc);
	}
	write_file(json_text(json), drafts_path());
}

} // namespace pvpcap
