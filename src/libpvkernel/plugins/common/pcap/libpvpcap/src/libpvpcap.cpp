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

#include "../include/libpvpcap.h"

#include <libpvpcap/shell.h>
#include <rapidjson/encodings.h>
#include <fstream>
#include <iterator>

#include "../include/libpvpcap/ws.h"
#include "../include/libpvpcap/exception.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

namespace pvpcap
{

/*******************************************************************************
 *
 * file_exists
 *
 ******************************************************************************/
bool file_exists(const std::string& file_name)
{
	std::ifstream input_file(file_name.c_str());
	return input_file.good();
}

std::string protocols_dict_path()
{
	return ws_protocols_dict_path();
}

void create_protocols_dict(std::string const& protocols_dict_file)
{
	return ws_create_protocols_dict(protocols_dict_file);
}

rapidjson::Document parse_protocol_dict(const std::string& protocols_dict_file)
{
	return ws_parse_protocol_dict(protocols_dict_file);
}

void enrich_protocols_tree(rapidjson::Document& enriched_protocols_tree,
                           const rapidjson::Document& protocols_tree,
                           const rapidjson::Document& protocols_dict)
{
	ws_enrich_protocols_tree(enriched_protocols_tree, protocols_tree, protocols_dict);
}

/*******************************************************************************
 *
 * build_json_document
 *
 ******************************************************************************/
rapidjson::Document create_protocols_tree(std::string const& source,
                                          const rapidjson::Document& protocols_dict)
{
	return ws_create_protocols_tree(source, protocols_dict);
}

void merge_protocols_tree(rapidjson::Value& dst,
                          const rapidjson::Value& src,
                          rapidjson::Document::AllocatorType& allocator)
{
	ws_merge_protocols_tree(dst, src, allocator);
}

/*******************************************************************************
 *
 * save_profile_data
 *
 ******************************************************************************/
void save_profile_data(rapidjson::Document& json_data, std::string const& profile_path)
{
	// save json to file
	rapidjson::StringBuffer str_buffer;
	rapidjson::Writer<rapidjson::StringBuffer> writer(str_buffer);
	json_data.Accept(writer);

	std::ofstream ofs(profile_path);
	ofs << str_buffer.GetString();

	if (!ofs.good())
		throw PcapTreeException("Can't write the JSON string to the file!");
}

/*******************************************************************************
 *
 * load_profile_data
 *
 ******************************************************************************/
void load_profile_data(rapidjson::Document& json_data, std::string const& profile_path)
{
	// Open json file
	std::ifstream ifs(profile_path);

	if (not ifs)
		throw PcapTreeException("Unable to open profile json file");

	std::string content{std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>()};

	// json DOM
	json_data.SetObject();
	json_data.Parse(content.c_str());

	if (json_data.HasParseError())
		throw PcapTreeException("profile json parse error");
}

/*******************************************************************************
 *
 * create_profile
 *
 ******************************************************************************/
void create_profile(std::string const& profile)
{
	rapidjson::Document json_data;

	// initialize the json document
	json_data.SetObject();
	rapidjson::Document::AllocatorType& alloc = json_data.GetAllocator();

	rapidjson::Value val;

	// add version attribute
	val.SetString("", alloc);
	json_data.AddMember("tshark_version", val, alloc);

	// add source attribute
	val.SetString("", alloc);
	json_data.AddMember("source", val, alloc);

	// add read only attribute
	val.SetBool(false);
	json_data.AddMember("read_only", val, alloc);

	// add options attribute
	json_data.AddMember("options", rapidjson::Value(rapidjson::kObjectType), alloc);

	// add options fields with default values
	val.SetBool(false);
	json_data["options"].AddMember("source", val, alloc);

	val.SetBool(false);
	json_data["options"].AddMember("destination", val, alloc);

	val.SetBool(false);
	json_data["options"].AddMember("protocol", val, alloc);

	val.SetBool(false);
	json_data["options"].AddMember("info", val, alloc);

	// filters
	val.SetString("", alloc);
	json_data["options"].AddMember("filters", val, alloc);

	// rewrite options
	val.SetBool(false);
	json_data["options"].AddMember("header", val, alloc);

	val.SetString("|", alloc);
	json_data["options"].AddMember("aggregator", val, alloc);

	val.SetString("l", alloc);
	json_data["options"].AddMember("occurrence", val, alloc);

	// add empty list of children
	json_data.AddMember("children", rapidjson::Value(rapidjson::kArrayType), alloc);

	// save json to file
	save_profile_data(json_data, get_user_profile_path(profile));
}

/*******************************************************************************
 *
 * get_user_profile_list
 *
 ******************************************************************************/
std::vector<std::string> get_user_profile_list()
{
	return get_directory_files(get_user_profile_dir());
}

std::vector<std::string> get_system_profile_list()
{
	return get_directory_files(get_system_profile_dir());
}

} // namespace pvpcap
