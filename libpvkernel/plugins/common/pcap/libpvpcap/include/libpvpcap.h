/*!
 * \file
 * \brief Manage the tree of the protocols in pcap.
 *
 * Manage the tree of the protocols extracted from
 * a packet capture file (pcap/pcapng).
 *
 * \copyright (C) Philippe Saade - Justin Teliet 2014
 * \copyright (C) ESI Group INENDI 2017
 */

#ifndef LIBPVPCAP_H
#define LIBPVPCAP_H

#include "rapidjson/document.h"

#include <vector>

/*!
 * Namespace that groups together the components of managing
 * the tree of protocols in a pcap file.
 */
namespace pvpcap
{

// use BEL character as separator as it is very unlikely to be present
static constexpr const char SEPARATOR[] = "\a";
static constexpr const char QUOTE[] = "n";

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
