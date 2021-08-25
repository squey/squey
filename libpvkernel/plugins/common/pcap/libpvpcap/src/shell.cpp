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
#include "../include/libpvpcap/ws.h"

#include <atomic>
#include <numeric>
#include <thread>
#include <unordered_set>
#include <mutex>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <signal.h>
#include <pwd.h>

#include <pvkernel/core/PVConfig.h>
#include <QDir>

#include <tbb/pipeline.h>

#include <string_view>

extern char** environ;

namespace pvpcap
{

/*******************************************************************************
 *
 * execute_cmd
 *
 ******************************************************************************/
std::vector<std::string> execute_cmd(const std::string cmd)
{
	// TODO: cmd size limite

	FILE* output;
	char buffer[1024];               // 1024/2048/3072/4096/5120/10240/
	std::vector<std::string> result; // = new std::vector<std::string>;

	std::cout << cmd << std::endl;

	if (!(output = popen(cmd.c_str(), "r"))) {
		std::cerr << "Error: can't execute popen command..." << std::endl;
		exit(EXIT_FAILURE);
	}

	while (fgets(buffer, sizeof(buffer), output) != NULL) {
		result.push_back(buffer);
	}

	if (pclose(output) != 0) { // en cas d'erreur
		std::cerr << "Error: can't execute pclose command..." << std::endl;
		exit(EXIT_FAILURE);
	}

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

	std::unordered_set<pid_t> pids;
	std::mutex pids_mutex;

	char* cmd_opts[cmd.size() + 2];
	for (size_t i = 0; i < cmd.size(); i++) {
		cmd_opts[i] = const_cast<char*>(cmd.at(i).c_str());
	}
	cmd_opts[cmd.size()] = nullptr;

	// Override XDG_CONFIG_HOME variable to find wireshark profiles
	const char* homedir;
	if ((homedir = getenv("HOME")) == NULL) {
		homedir = getpwuid(getuid())->pw_dir;
	}
	std::basic_string<char*> env_vars(environ);
	std::string xdg_config_home("XDG_CONFIG_HOME=");
	auto it = std::find_if(
	    env_vars.begin(), env_vars.end(), [&xdg_config_home](std::string_view env_var) {
		    return env_var.substr(0, xdg_config_home.size()).compare(xdg_config_home) == 0;
		});
	assert(it != env_vars.end());
	std::string xdg_config_home_value = get_wireshark_profiles_dir();
	*it = xdg_config_home_value.data();

	const size_t max_number_of_live_token =
	    std::min(files.size(), (size_t)std::thread::hardware_concurrency());

	size_t file_count = 0;
	tbb::parallel_pipeline(
	    max_number_of_live_token,
	    tbb::make_filter<void, splitted_file_t>(
	        tbb::filter::serial_in_order,
	        [&](tbb::flow_control& fc) {

		        if (file_count < files.size() and not canceled) {
			        filenames.emplace_back(files[file_count]);
			        filenames.back().path() += ".csv";

			        return files[file_count++];
		        } else {
			        fc.stop();

			        if (canceled) { // force running processes to terminate and remove pcap files
				        for (const pid_t& pid : pids) {
					        kill(pid, 15);
				        }
				        for (const splitted_file_t& pcap_file : files) {
					        std::remove(pcap_file.path().c_str());
				        }
			        }

			        return splitted_file_t();
		        }
		    }) &
	        tbb::make_filter<splitted_file_t, void>(
	            tbb::filter::parallel, [&](splitted_file_t pcap) {

		            if (not pcap.path().empty()) {
			            std::string csv_path = pcap.path() + ".csv";

			            /**
			             * using 'vfork' instead of 'fork' because of some very serious
			             * performance problems related to the 'fork' implementation
			             */
			            pid_t pid = vfork();
			            if (pid == 0) {
				            int fd_in = open(pcap.path().c_str(), O_RDONLY, O_CLOEXEC, 0);
				            int fd_out = open(csv_path.c_str(),
				                              O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0666);
				            dup2(fd_out, STDOUT_FILENO);
				            dup2(fd_in, STDIN_FILENO);
				            if (execvpe(cmd_opts[0], cmd_opts, env_vars.data()) == -1) {
					            _exit(-1);
				            }
			            } else {
				            pids_mutex.lock();
				            pids.emplace(pid);
				            pids_mutex.unlock();

				            int status = 0;
				            waitpid(pid, &status, 0);

				            pids_mutex.lock();
				            pids.erase(pid);
				            pids_mutex.unlock();

				            if (WIFEXITED(status) and WEXITSTATUS(status) == 0) {
					            if (f_progression) {
						            processed_pcap_packets_count += pcap.packets_count();
						            f_progression(processed_pcap_packets_count);
					            }
				            }

				            // remove pcap file
				            std::remove(pcap.path().c_str());
			            }
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
void save_to_file(std::string const file_name, std::vector<std::string> const& text_to_save)
{
	std::cout << "Save to " << file_name << "..." << std::endl;

	std::ofstream output_file(file_name);

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
bool is_directory(std::string path_name)
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
std::vector<std::string> get_directory_files(std::string path_name)
{
	std::vector<std::string> files; // = new std::vector<std::string>();
	DIR* dir;
	struct dirent* ent;
	if ((dir = opendir(path_name.c_str())) != NULL) {
		/* get only all the files */
		while ((ent = readdir(dir)) != NULL) {
			if (!is_directory(path_name + "/" + ent->d_name)) {
				files.push_back(ent->d_name);
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
std::string get_file_extension(std::string file_name)
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
	QString user_conf_dir = QString::fromStdString(PVCore::PVConfig::user_dir()) +
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
	QString pluginsdirs = QString(getenv("PVKERNEL_PLUGIN_PATH"));

	if (pluginsdirs.isEmpty()) {
		return (QString(PVKERNEL_PLUGIN_PATH) + "/input-types/pcap/profiles").toStdString();
	} else {
		return "../libpvkernel/plugins/common/pcap/profiles";
	}
}

} // namespace pvpcap
