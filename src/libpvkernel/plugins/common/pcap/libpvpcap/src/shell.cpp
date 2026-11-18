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

#include "../include/libpvpcap/shell.h"

#include <sys/stat.h>
#include <pvkernel/core/PVConfig.h>
#include <dirent.h>
#include <qchar.h>
#include <qstring.h>
#include <stdlib.h>
#include <string.h>
#include <cstring>
#include <unistd.h>
#include <atomic>
#include <numeric>
#include <thread>
#include <unordered_set>
#include <mutex>
#include <csignal>
#include <QDir>
#include <string_view>
#include <algorithm>
#include <compare>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>

#include "../include/libpvpcap/ws.h"
#include <pvbase/general.h> // IWYU pragma: keep

#include <boost/dll/runtime_symbol_info.hpp>

namespace pvpcap
{

/*******************************************************************************
 *
 * is_directory
 *
 ******************************************************************************/
bool is_directory(const std::string& path_name)
{
	struct stat info;

	if (stat(path_name.c_str(), &info) != 0) {
		std::cerr << "cannot access " << path_name << std::endl;
		return false;
	} else if (info.st_mode & S_IFDIR)
		return true;
	else
		return false;
}

/*******************************************************************************
 *
 * get_directory_files
 *
 ******************************************************************************/
std::vector<std::string> get_directory_files(const std::string& path_name)
{
	std::vector<std::string> files; // = new std::vector<std::string>();
	DIR* dir;
	struct dirent* ent;
	if ((dir = opendir(path_name.c_str())) != nullptr) {
		/* get only all the files */
		while ((ent = readdir(dir)) != nullptr) {
			if (!is_directory(path_name + "/" + ent->d_name)) {
				files.emplace_back(ent->d_name);
			}
		}
		closedir(dir);
	} else {
		std::cerr << "Error: could not open directory \"" << path_name << "\"" << std::endl;
	}

	return files;
}

/*******************************************************************************
 *
 * get_user_conf_dir
 *
 ******************************************************************************/
std::string get_user_conf_dir()
{
	// FIXME: How to manage standalone version
	QString user_conf_dir = PVCore::PVConfig::user_dir() +
	                        QDir::separator() + "plugins" + QDir::separator() + "pcapsicum";

	// we should do this at install time
	// create if not exists
	if (not QDir().mkpath(user_conf_dir)) {
		/* Report and carry on, as get_user_profile_dir() below does. */
		std::cerr << "Can't create user configuration directory: " << user_conf_dir.toStdString()
		          << std::endl;
	}

	return user_conf_dir.toStdString();
}

/*******************************************************************************
 *
 * get_user_profile_dir
 *
 ******************************************************************************/
std::string get_user_profile_dir()
{
	// FIXME: How to manage standalone version
	QString user_profile_dir =
	    QString::fromStdString(get_user_conf_dir()) + QDir::separator() + "profiles";

	// we should do this at install time
	// create if not exists
	if (not QDir().mkpath(user_profile_dir)) {
		std::cerr << "Can't create user profile directory: " << user_profile_dir.toStdString()
		          << std::endl;
	}

	return user_profile_dir.toStdString();
}

std::string get_user_profile_path(const std::string& filename)
{
	return get_user_profile_dir() + "/" + filename;
}

/*******************************************************************************
 *
 * get_system_profile_dir
 *
 ******************************************************************************/
std::string get_system_profile_dir()
{
	QString pcap_profiles_dir = QString(getenv("SQUEY_PCAP_PROFILES_PATH"));

	if (not pcap_profiles_dir.isEmpty()) {
		return pcap_profiles_dir.toStdString();
	}
	else {
#ifndef __linux__
		boost::filesystem::path exe_path = boost::dll::program_location();
#endif
#ifdef __APPLE__
		QString pluginsdirs = QString::fromStdString(exe_path.parent_path().string() + "/../PlugIns");
#elifdef _WIN32
		QString pluginsdirs = QString::fromStdString(exe_path.parent_path().string() + "/plugins");
#else
		QString pluginsdirs = QString(PVKERNEL_PLUGIN_PATH);
#endif
		return (pluginsdirs + "/input-types/pcap/profiles").toStdString();
	}
}

} // namespace pvpcap
