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

#ifndef SHELL_H
#define SHELL_H

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <limits.h>
#include <unistd.h>
#include <boost/algorithm/string.hpp> // split
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <iostream>
#include <vector>
#include <functional>
#include <string>

/*!
 * Namespace that groups together the components of managing
 * the tree of protocols in a pcap file.
 */
namespace pvpcap
{

/**
 * The files of a directory, by name, hidden ones aside.
 */
std::vector<std::string> get_directory_files(const std::string& path_name);

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
