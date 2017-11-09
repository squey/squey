/*!
 * \file
 * \brief Interface to manage the shell commands.
 *
 * All the shell commands are dealt here.
 *
 * \copyright (C) Philippe Saade - Justin Teliet 2014
 * \copyright (C) ESI Group INENDI 2017
 */

#ifndef SHELL_H
#define SHELL_H

#include <iostream>
#include <vector>

#include <sys/types.h>
#include <sys/stat.h>

#include <dirent.h>

#include <limits.h>
#include <unistd.h>

#include <boost/algorithm/string.hpp> // split
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include "pcap_splitter.h"

/*!
 * Namespace that groups together the components of managing
 * the tree of protocols in a pcap file.
 */
namespace pvpcap
{

/**
 * Execute a given command.
 *
 * @param cmd command to execute (absolute path).
 *
 * @return the result of the command.
 */
std::vector<std::string> execute_cmd(const std::string cmd);

splitted_files_t
extract_csv(splitted_files_t files,
            const std::vector<std::string>& cmd,
            bool& canceled,
            const std::function<void(size_t total_datasize)>& f_total_datasize = {},
            const std::function<void(size_t current_datasize)>& f_progression = {});

/**
 * Save to file.
 *
 * @param file_name output file name.
 * @param text_to_save text to save.
 *
 */
void save_to_file(std::string const file_name, std::vector<std::string> const& text_to_save);

/**
 * Test if it is directory.
 *
 * @param path_name path name.
 * @return true or false
 *
 */
bool is_directory(std::string path_name);

/**
 * Return a list of files within directory.
 *
 * @param path_name path name.
 * @return list of file
 *
 */
std::vector<std::string> get_directory_files(std::string path_name);

/**
 * Return a file extension.
 *
 * @param file_name file name.
 * @return file extension
 *
 */
std::string get_file_extension(std::string file_name);

/**
 * Return the user configuration directory for pcapsicum.
 *
 * @return user configuration directory.
 *
 */
std::string get_user_conf_dir();

/**
 * Return the user profile directory for pcapsicum.
 *
 * @return user profile directory.
 *
 */
std::string get_user_profile_dir();
std::string get_user_profile_path(const std::string& filename);

/**
 * Return the system profile directory for pcapsicum.
 *
 * @return system profile directory.
 *
 */
std::string get_system_profile_dir();

} /* namespace pvpcap */

#endif // SHELL_H
