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

#include <pvkernel/core/PVUtils.h>
#include <stdio.h>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/detail/classification.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <fstream>
#include <memory>
#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <QTemporaryDir>
#include <QTemporaryFile>

#ifdef _WIN32
#include <windows.h>
#endif

#include <pvlogger.h>

std::string& PVCore::replace(std::string& str,
                             const std::string& from,
                             const std::string& to,
                             size_t pos /*= 0*/)
{
	// avoid an infinite loop if "from" is an empty string
	if (from.empty()) {
		return str;
	}

	while ((pos = str.find(from, pos)) != std::string::npos) {
		str.replace(pos, from.size(), to);
		// Advance to avoid replacing the same substring again
		pos += to.size();
	}

	return str;
}

std::string PVCore::file_content(const std::string& file_path)
{
	std::ifstream stream(std::filesystem::path{file_path});

	return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::string PVCore::exec_cmd(const char* cmd)
{
	static constexpr const size_t MAX_LINE_SIZE = 128;

	std::array<char, MAX_LINE_SIZE> buffer;
	std::string result;
	std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
	if (not pipe)
		throw std::runtime_error(std::string("running command '") + std::string(cmd) +
		                         "'  failed!");
	while (not feof(pipe.get())) {
		if (fgets(buffer.data(), MAX_LINE_SIZE, pipe.get()) != nullptr) {
			result += buffer.data();
		}
	}
	return result;
}

void PVCore::remove_common_folders(std::vector<std::string>& paths)
 {
	std::vector<std::vector<std::string>> paths_folders;
	for (std::string& path : paths) {
		paths_folders.emplace_back(std::vector<std::string>());
		boost::split(paths_folders.back(), path, boost::is_any_of("/"));
	}

	size_t common_folders_depth = 0;
	bool stop = false;
	do {
		std::string current_folder;
		for (const std::vector<std::string>& path_folders : paths_folders) {
			if (current_folder.empty()) {
				current_folder = path_folders[common_folders_depth];
			}
			stop |= common_folders_depth >= path_folders.size()-1 or path_folders[common_folders_depth] != current_folder;
		}
		common_folders_depth += not stop;
	} while (not stop);
	common_folders_depth--;

	if (common_folders_depth > 0) {
		for (std::vector<std::string>& path_folders : paths_folders) {
			path_folders.erase(path_folders.begin(), path_folders.size() > common_folders_depth+1 ? path_folders.begin() + common_folders_depth+1 : path_folders.end());
		}
		for (size_t i = 0; i < paths.size(); i++) {
			paths[i] = boost::join(paths_folders[i], std::string("/"));
		}
	}
}

QString PVCore::mkdtemp(QString tmpl)
{
	QTemporaryDir tmp_dir(tmpl);
	tmp_dir.setAutoRemove(false);

	if (!tmp_dir.isValid()) {
		pvlogger::error() << "Failed to create temporary directory '" << qPrintable(tmpl) << "' : " << qPrintable(tmp_dir.errorString()) << std::endl;
		return {};
	}

	return tmp_dir.path();
}

QString PVCore::mkstemp(QString tmpl)
{
	QTemporaryFile tmp_file(tmpl);
	tmp_file.setAutoRemove(false);

	if (!tmp_file.open()) {
		return {};
	}
	tmp_file.close();

	return tmp_file.fileName();
}

#ifdef __linux__

size_t PVCore::available_memory()
{
    std::string token;
    std::ifstream file("/proc/meminfo");
    while(file >> token) {
        if(token == "MemAvailable:") {
            unsigned long mem;
            if(file >> mem) {
                return mem * 1024;
            } else {
                return 0;
            }
        }
        // Ignore the rest of the line
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    return 0; // Nothing found
}
#elifdef __APPLE__

#include <sys/types.h>
#include <sys/sysctl.h>

size_t PVCore::available_memory()
{
	int mib[2] = { CTL_HW, HW_MEMSIZE };
    uint64_t mem = 0;
    size_t len = sizeof(mem);

    if (sysctl(mib, 2, &mem, &len, nullptr, 0) == 0) {
        return mem * 1024;
    } else {
        perror("sysctl");
        return 0;
    }
}
#elif _WIN32
#include <windows.h>

size_t PVCore::available_memory()
{
    MEMORYSTATUSEX memStatus;
    memStatus.dwLength = sizeof(MEMORYSTATUSEX);

    if (GlobalMemoryStatusEx(&memStatus)) {
        return static_cast<size_t>(memStatus.ullAvailPhys);
    }
	else {
        return 0;
    }
}
#else
	static_assert(false, "__func__ not supported for this target.");
#endif


#if __APPLE__

#include <iostream>
#include <string>
#include <vector>
#include <sys/sysctl.h>

int PVCore::process_running_count(const std::string& process_name) {
	int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_ALL, 0};
    size_t buffer_size = 0;

    // Get the size of the buffer needed to store process information
    if (sysctl(mib, 4, nullptr, &buffer_size, nullptr, 0) == -1) {
        perror("sysctl error");
        return 0;
    }

    // Allocate the buffer and retrieve process information
    std::vector<char> buffer(buffer_size);
    if (sysctl(mib, 4, buffer.data(), &buffer_size, nullptr, 0) == -1) {
        perror("sysctl error");
        return 0;
    }

    // Iterate through processes to count matching names
    struct kinfo_proc* process_list = reinterpret_cast<struct kinfo_proc*>(buffer.data());
    size_t process_count = buffer_size / sizeof(struct kinfo_proc);
    int instance_count = 0;

    for (size_t i = 0; i < process_count; i++) {
        std::string name = process_list[i].kp_proc.p_comm;
        if (name == process_name) {
            instance_count++;
        }
    }

    return instance_count;
}
#endif

int PVCore::setenv(const char* name, const char* value, int overwrite)
{
#ifdef _WIN32
    int wname_len = MultiByteToWideChar(CP_UTF8, 0, name, -1, nullptr, 0);
    if (wname_len == 0) return -1;
    std::wstring wname(wname_len, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, name, -1, &wname[0], wname_len);

    if (!overwrite and _wgetenv(wname.c_str()) != nullptr) {
        return 0;
    }

    int wvalue_len = MultiByteToWideChar(CP_UTF8, 0, value, -1, nullptr, 0);
    if (wvalue_len == 0) {
		return -1;
	}

    std::wstring wvalue(wvalue_len, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, value, -1, &wvalue[0], wvalue_len);

    return _wputenv_s(wname.c_str(), wvalue.c_str());
#else
	return ::setenv(name, value, overwrite);
#endif
}

char* PVCore::getenv(const char* env_var)
{
#ifdef _WIN32
    int wenv_var_len = MultiByteToWideChar(CP_UTF8, 0, env_var, -1, nullptr, 0);
    if (wenv_var_len == 0) {
        return nullptr;
    }

    std::wstring wenv_var(wenv_var_len, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, env_var, -1, &wenv_var[0], wenv_var_len);

    const wchar_t* wvalue = _wgetenv(wenv_var.c_str());
    if (not wvalue) {
        return nullptr;
    }

    int utf8_len = WideCharToMultiByte(CP_UTF8, 0, wvalue, -1, nullptr, 0, nullptr, nullptr);
    if (utf8_len == 0) {
        return nullptr;
    }

    static thread_local std::string utf8_value;
    utf8_value.resize(utf8_len - 1);
    WideCharToMultiByte(CP_UTF8, 0, wvalue, -1, &utf8_value[0], utf8_len, nullptr, nullptr);

    return const_cast<char*>(utf8_value.c_str());
#else
	return std::getenv(env_var);
#endif
}