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

#ifndef LIBPVPCAP_H
#define LIBPVPCAP_H

#include <rapidjson/allocators.h>
#include <rapidjson/rapidjson.h>
#include <vector>
#include <string>
#include <QDir>
#include <QStandardPaths>

#include "rapidjson/document.h"

/*!
 * Namespace that groups together the components of managing
 * the tree of protocols in a pcap file.
 */
namespace pvpcap
{

// use DEL character as separator as it is very unlikely to be present and is XML 1.0 compliant
static constexpr const char SEPARATOR[] = "\x7F";
static constexpr const char QUOTE[] = "n";


std::string tshark_path();

/**
 * Test if a file exist.
 *
 * @param file_name absolute path to the file.
 *
 * @return true if file exist.
 */
bool file_exists(const std::string& file_name);

std::string protocols_dict_path();

void create_protocols_dict(std::string const& protocols_dict_file);

rapidjson::Document parse_protocol_dict(const std::string& protocols_dict_file);

void enrich_protocols_tree(rapidjson::Document& enriched_protocols_tree,
                           const rapidjson::Document& protocols_tree,
                           const rapidjson::Document& protocols_dict);

/**
 * Build json tree structure from a pcap file.
 *
 * @param source pcap file name.
 * @param  document rapidjson document.
 */
rapidjson::Document create_protocols_tree(std::string const& source,
                                          const rapidjson::Document& protocols_dict);

void merge_protocols_tree(rapidjson::Value& dst,
                          const rapidjson::Value& src,
                          rapidjson::Document::AllocatorType& allocator);

/**
 * Get all selected field grouped by protocol.
 * Avoid all duplicated fields (filter_name is unique)
 *
 * @param  json_data rapidjson document of pcap file tree.
 * @return rapidjson::Value* tree of selected fields
 */
void overview_selected_fields(rapidjson::Document& json_data, rapidjson::Document& overview_data);

/**
 *
 * @param json_data
 */
void save_profile_data(rapidjson::Document& json_data, std::string const& profile_path);

/**
 *
 * @param json_data
 */
void load_profile_data(rapidjson::Document& json_data, std::string const& profile_path);

/**
 *
 * @param filename
 */
void create_profile(std::string const& profile);

/**
 *
 */
std::vector<std::string> get_user_profile_list();
std::vector<std::string> get_system_profile_list();

} // namespace pvpcap

#endif // LIBPVPCAP_H
