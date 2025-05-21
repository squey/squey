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

#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <windows.h>
#include <process.h>
#else
#include <spawn.h>
#include <sys/wait.h>
#include <pwd.h>
#endif
#include <pvkernel/core/PVConfig.h>
#include <tbb/parallel_pipeline.h>
#include <dirent.h>
#include <libpvpcap/pcap_splitter.h>
#include <qchar.h>
#include <qstring.h>
#include <stdlib.h>
#include <string.h>
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

#if __APPLE__
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

int execvpe(const char *file, char *const argv[], char *const envp[]) {
    // Check if the path is absolute or relative
    if (strchr(file, '/') != NULL) {
        // Use execve directly if the path is given
        return execve(file, argv, envp);
    }

    // Get the PATH from the environment
    const char *path = getenv("PATH");
    if (!path) {
        errno = ENOENT; // PATH not set
        return -1;
    }

    // Copy PATH to avoid modifying the original
    char *path_copy = strdup(path);
    if (!path_copy) {
        return -1; // Allocation failed
    }

    char *saveptr;
    char *dir = strtok_r(path_copy, ":", &saveptr);

    while (dir != NULL) {
        // Build the full path to the file
        char full_path[PATH_MAX];
        snprintf(full_path, sizeof(full_path), "%s/%s", dir, file);

        // Attempt to execute with execve
        execve(full_path, argv, envp);

        // If execve fails, move to the next directory in PATH
        if (errno != ENOENT) {
            free(path_copy);
            return -1; // Failure for reasons other than file not found
        }

        dir = strtok_r(NULL, ":", &saveptr);
    }

    // Free memory and set errno
    free(path_copy);
    errno = ENOENT;
    return -1;
}
#endif

extern char** environ;

namespace pvpcap
{

/*******************************************************************************
 *
 * execute_cmd
 *
 ******************************************************************************/
std::vector<std::string> execute_cmd(const std::string& cmd)
{
	// TODO: cmd size limite

	FILE* output;
	char buffer[1024];
	std::vector<std::string> result;

#ifdef _WIN32
	std::string cmd_wrapper = std::string("cmd.exe /C ") + cmd;
	if (!(output = _popen(cmd_wrapper.c_str(), "r"))) {
#else
	if (!(output = popen(cmd.c_str(), "r"))) {
#endif
		pvlogger::error() << "Can't execute '" << cmd << "'" << std::endl;
	}

	while (fgets(buffer, sizeof(buffer), output) != nullptr) {
		result.emplace_back(buffer);
	}

#ifdef _WIN32
	_pclose(output);
#else
	pclose(output);
#endif

	return result;
}

splitted_files_t
extract_csv(splitted_files_t files,
            const std::vector<std::string>& cmd,
            bool& canceled,
            const std::function<void(size_t total_datasize)>& f_total_datasize /* = {} */,
            const std::function<void(size_t current_datasize)>& f_progression /* = {} */)
{
	splitted_files_t filenames;

	std::atomic<size_t> processed_pcap_packets_count(0);
	size_t total_packets_count =
	    std::accumulate(files.begin(), files.end(), 0UL,
	                    [](size_t t, const splitted_file_t& v) { return t + v.packets_count(); });
	if (f_total_datasize) {
		f_total_datasize(total_packets_count);
	}

	std::unordered_set<
#ifdef _WIN32
	HANDLE
#else
	pid_t
#endif
	> pids;
	std::mutex pids_mutex;

	std::vector<char*> cmd_opts(cmd.size() + 2);
	for (size_t i = 0; i < cmd.size(); i++) {
		cmd_opts[i] = const_cast<char*>(cmd.at(i).c_str());
	}
	cmd_opts[cmd.size()] = nullptr;

	// Override XDG_CONFIG_HOME variable to find wireshark profiles
	std::basic_string<char*> env_vars(environ);
	std::string xdg_config_home("XDG_CONFIG_HOME=");
	auto it = std::find_if(
	    env_vars.begin(), env_vars.end(), [&xdg_config_home](std::string_view env_var) {
		    return env_var.substr(0, xdg_config_home.size()).compare(xdg_config_home) == 0;
		});
	if (it != env_vars.end()) {
		std::string xdg_config_home_value = get_wireshark_profiles_dir();
		*it = xdg_config_home_value.data();
	}

	const size_t max_number_of_live_token =
	    std::min(files.size(), (size_t)std::thread::hardware_concurrency());

	size_t file_count = 0;
	tbb::parallel_pipeline(
	    max_number_of_live_token,
	    tbb::make_filter<void, splitted_file_t>(
	        tbb::filter_mode::serial_in_order,
	        [&](tbb::flow_control& fc) {

		        if (file_count < files.size() and not canceled) {
			        filenames.emplace_back(files[file_count]);
			        filenames.back().path() += ".csv";

			        return files[file_count++];
		        } else {
			        fc.stop();

			        if (canceled) { // force running processes to terminate and remove pcap files
				        for (const auto& pid : pids) {
#ifdef _WIN32
							TerminateProcess(pid, 1);
#else
					        kill(pid, 15);
#endif
				        }
				        // for (const splitted_file_t& pcap_file : files) {
					    //     std::remove(pcap_file.path().c_str());
				        // }
			        }

			        return splitted_file_t();
		        }
		    }) &
	        tbb::make_filter<splitted_file_t, void>(
	            tbb::filter_mode::parallel, [&](splitted_file_t pcap) {

		            if (not pcap.path().empty()) {
			            std::string csv_path = pcap.path() + ".csv";

#ifdef _WIN32
						STARTUPINFO si{};
						PROCESS_INFORMATION pi{};
						si.dwFlags = STARTF_USESTDHANDLES;
						SECURITY_ATTRIBUTES sa{};
						sa.nLength = sizeof(SECURITY_ATTRIBUTES);
						sa.lpSecurityDescriptor = NULL;
						sa.bInheritHandle = TRUE;
						std::wstring pcap_wpath = std::filesystem::path(pcap.path()).wstring();
						HANDLE hStdin = CreateFileW(pcap_wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, &sa, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
						if (hStdin == INVALID_HANDLE_VALUE) {
							pvlogger::error() << "Failed to open input file: " << GetLastError() << std::endl;
						}
						std::wstring csv_wpath = std::filesystem::path(csv_path).wstring();
						HANDLE hStdout = CreateFileW(csv_wpath.c_str(), GENERIC_WRITE, 0, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
						if (hStdout == INVALID_HANDLE_VALUE) {
							pvlogger::error() << "Failed to open output file: " << GetLastError() << std::endl;
						}

						si.cb = sizeof(STARTUPINFOA);
						si.hStdInput = hStdin;
						si.hStdOutput = hStdout;
						si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
						si.dwFlags |= STARTF_USESTDHANDLES;

						std::string cmdline = boost::algorithm::join(cmd, " ");
						std::wstring wcmdline = QString::fromStdString(cmdline).toStdWString();
						BOOL status = CreateProcessW(
							nullptr,
							wcmdline.data(),
							nullptr,
							nullptr,
							TRUE,
							CREATE_UNICODE_ENVIRONMENT | CREATE_NEW_PROCESS_GROUP,
							/*env_vars.data()*/ nullptr, // FIXME
							nullptr,
							&si,
							&pi
						);
						if (not status) {
							LPVOID error_buffer;
							FormatMessageW(
								FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
								NULL,
								GetLastError(),
								MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
								(LPWSTR)&error_buffer,
								0,
								NULL);
							pvlogger::error() << "Unable to execute '" << cmdline << "' : " << (char*)error_buffer << std::endl;
							LocalFree(error_buffer);
						}

						pids_mutex.lock();
						pids.emplace(pi.hProcess);
						pids_mutex.unlock();

						WaitForSingleObject(pi.hProcess, INFINITE);
						DWORD exit_code = -1;
						GetExitCodeProcess(pi.hProcess, &exit_code);

						pids_mutex.lock();
						pids.erase(pi.hProcess);
						pids_mutex.unlock();

						if (exit_code == 0) {
							if (f_progression) {
								processed_pcap_packets_count += pcap.packets_count();
						        f_progression(processed_pcap_packets_count);
							}
						}
						else if (status) {
							pvlogger::error() << "'" << cmdline << "' exit code: " << exit_code << std::endl;
						}
						CloseHandle(hStdin);
						CloseHandle(hStdout);
						CloseHandle(pi.hProcess);
						CloseHandle(pi.hThread);

						// remove pcap file
				        std::remove(pcap.path().c_str());

#else
						posix_spawn_file_actions_t actions;
						posix_spawn_file_actions_init(&actions);

						// Set up file descriptors
						int fd_in = open(pcap.path().c_str(), O_RDONLY | O_CLOEXEC);
						int fd_out = open(csv_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0666);

						if (fd_in == -1 || fd_out == -1) {
							perror("open failed");
							if (fd_in != -1) close(fd_in);
							if (fd_out != -1) close(fd_out);
							return;
						}

						posix_spawn_file_actions_adddup2(&actions, fd_out, STDOUT_FILENO);
						posix_spawn_file_actions_adddup2(&actions, fd_in, STDIN_FILENO);

						pid_t pid = 0;
						int status = posix_spawnp(
							&pid,
							cmd_opts[0],
							&actions,
							nullptr,
							cmd_opts.data(),
							env_vars.data()
						);

						posix_spawn_file_actions_destroy(&actions);
						close(fd_in);
						close(fd_out);

						if (status != 0) {
							perror("posix_spawnp failed");
							return;
						}

						{
							std::lock_guard<std::mutex> lock(pids_mutex);
							pids.emplace(pid);
						}

						int exit_status = 0;
						waitpid(pid, &exit_status, 0);

						{
							std::lock_guard<std::mutex> lock(pids_mutex);
							pids.erase(pid);
						}

						if (WIFEXITED(exit_status) && WEXITSTATUS(exit_status) == 0) {
							if (f_progression) {
								processed_pcap_packets_count += pcap.packets_count();
								f_progression(processed_pcap_packets_count);
							}
						}

						// remove pcap file
						std::remove(pcap.path().c_str());
#endif
		            }
		        }));

	// regroup tmp csv files by original pcap file as this will simplify pcap export afterwards
	std::sort(filenames.begin(), filenames.end(), [](const auto& f1, const auto& f2) {
		return f1.original_pcap_path() < f2.original_pcap_path();
	});

	return filenames;
}

/*******************************************************************************
 *
 * save_to_file
 *
 ******************************************************************************/
void save_to_file(const std::string& file_name, std::vector<std::string> const& text_to_save)
{
	std::cout << "Save to " << file_name << "..." << std::endl;

	std::ofstream output_file{std::filesystem::path(file_name)};

	if (!output_file) {
		std::cerr << "Error: can't open output file \"" << file_name << "\"" << std::endl;
		exit(EXIT_FAILURE);
	}

	std::ostream_iterator<std::string> output_iterator(output_file);
	std::copy(text_to_save.begin(), text_to_save.end(), output_iterator);
}

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
		exit(EXIT_FAILURE);
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
		/* could not open directory */
		std::cerr << "Error: could not open directory \"" << path_name << "\"" << std::endl;
		exit(EXIT_FAILURE);
	}

	return files;
}

/*******************************************************************************
 *
 * get_file_extension
 *
 ******************************************************************************/
std::string get_file_extension(const std::string& file_name)
{
	const char* ext = strchr(file_name.c_str(), '.');

	// no extentsion
	if (!ext)
		return "";

	// to eliminate hidden files
	if (sizeof(ext) == file_name.size())
		return "";
	return ext;
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
		std::cerr << "Can't create user configuration directory: " << user_conf_dir.toStdString()
		          << std::endl;
		exit(EXIT_FAILURE);
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
		exit(EXIT_FAILURE);
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
