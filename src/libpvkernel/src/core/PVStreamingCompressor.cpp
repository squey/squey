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

#include <pvkernel/core/PVStreamingCompressor.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#include <pvlogger.h>
#include <signal.h>
#include <stdlib.h>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/detail/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <cerrno>
#include <iostream>
#include <cstring> // for std::strerror
#include <cassert>
#include <algorithm>
#include <memory>

#include "pvkernel/core/PVOrderedMap.h"

#if __APPLE__
	const PVCore::PVOrderedMap<std::string, std::pair<std::string, std::string>>
    PVCore::__impl::PVStreamingBase::_supported_compressors = {};
#else
const PVCore::PVOrderedMap<std::string, std::pair<std::string, std::string>>
    PVCore::__impl::PVStreamingBase::_supported_compressors = {
        {"gz", {"pigz", "unpigz"}}, {"bz2", {"lbzip2", "lbunzip2"}}, {"zip", {"zip", "funzip"}}, {"xz", {"xz -T0", "unxz -T0"}}};
#endif

/******************************************************************************
 *
 * PVCore::PVStreamingBase
 *
 ******************************************************************************/

PVCore::__impl::PVStreamingBase::PVStreamingBase(const std::string& path)
    : _path(path)
    , _extension(path.substr(path.rfind('.') + 1))
    , _passthrough(_supported_compressors.find(_extension) == _supported_compressors.end())
{
}

PVCore::__impl::PVStreamingBase::~PVStreamingBase()
{
	close(_status_fd);
	_status_fd = -1;
}

void PVCore::__impl::PVStreamingBase::cancel()
{
	_canceled = true;
	wait_finished();
	_canceled = false;
}

void PVCore::__impl::PVStreamingBase::wait_finished()
{
	if (_finished) {
		return;
	}

	_finished = true;
	close(_fd);
	_fd = -1;

	if (_passthrough) {
		return;
	}

	do_wait_finished();
}

std::vector<std::string> PVCore::__impl::PVStreamingBase::supported_extensions()
{
	return _supported_compressors.keys();
}

std::tuple<std::vector<std::string>, std::vector<char*>> PVCore::__impl::PVStreamingBase::executable(const std::string& extension, EExecType type)
{
	std::string exec;
	auto it = _supported_compressors.find(extension);
	if (it != _supported_compressors.end()) {
		if (type == EExecType::COMPRESSOR) {
			exec = _supported_compressors.at(extension).first;
		}
		else {
			exec = _supported_compressors.at(extension).second;
		}
	}
	std::vector<std::string> args;
	boost::algorithm::split(args, exec, boost::is_any_of(" "));

	std::vector<char*> argv;
	for (const auto& arg : args) {
		argv.push_back((char*)arg.data());
	}
	argv.push_back(nullptr);

	return {args, argv};
}

int PVCore::__impl::PVStreamingBase::return_status(std::string* status_msg /* = nullptr */)
{
	if (_status_fd != -1 && status_msg != nullptr) {
		char buffer[1024];
		while (true) {
			int read_count = ::read(_status_fd, buffer, sizeof(buffer));
			if (read_count <= 0) {
				break;
			}
			_status_msg += std::string(buffer, 0, read_count);
		}

		close(_status_fd);
		_status_fd = -1;
	}

	if (status_msg != nullptr) {
		*status_msg = _status_msg;
	}

	return _status_code;
}

/******************************************************************************
 *
 * PVCore::PVStreamingCompressor
 *
 ******************************************************************************/

#include <fcntl.h>
#include <iostream>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern char **environ;

PVCore::PVStreamingCompressor::PVStreamingCompressor(const std::string& path)
    : __impl::PVStreamingBase(path)
{
    if (_passthrough) {
        _fd = open(path.c_str(), O_CREAT | O_WRONLY, 0666);
        if (_fd == -1) {
            throw PVCore::PVStreamingCompressorError("Unable to create file '" + path + "'");
        }
        return;
    }

    // Used to forward data to compressor
    int in_pipe[2];
    pipe(in_pipe);

    // Used to get error message back from compressor
    int out_pipe[2];
    pipe(out_pipe);

    // Used to check if execvp failed
    int exec_pipe[2];
    pipe(exec_pipe);

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);

    const auto& [args, argv] = executable(_extension, EExecType::COMPRESSOR);

    // Redirect stdin from in_pipe
    posix_spawn_file_actions_adddup2(&actions, in_pipe[STDIN_FILENO], STDIN_FILENO);
    close(in_pipe[STDOUT_FILENO]);

    // Redirect stderr to out_pipe
    posix_spawn_file_actions_adddup2(&actions, out_pipe[STDOUT_FILENO], STDERR_FILENO);
    close(out_pipe[STDIN_FILENO]);

    // Redirect stdout to the file
    int compression_fd = open(path.c_str(), O_CREAT | O_WRONLY | O_CLOEXEC, 0666);
    if (compression_fd == -1) {
        std::cerr << "Unable to create file '" + path + "'" << std::endl;
        posix_spawn_file_actions_destroy(&actions);
        throw PVCore::PVStreamingCompressorError("Unable to create file '" + path + "'");
    }
    posix_spawn_file_actions_adddup2(&actions, compression_fd, STDOUT_FILENO);
    close(compression_fd);

    // Mark exec_pipe[STDOUT_FILENO] to close-on-exec for error checking
    fcntl(exec_pipe[STDOUT_FILENO], F_SETFD, FD_CLOEXEC);

    pid_t child_pid;
    if (posix_spawn(&child_pid, args[0].c_str(), &actions, nullptr, argv.data(), environ) != 0) {
        posix_spawn_file_actions_destroy(&actions);
        throw PVCore::PVStreamingCompressorError("Failed to spawn compression process");
    }

    // Cleanup posix_spawn actions
    posix_spawn_file_actions_destroy(&actions);

    // Parent process setup
    setpgid(child_pid, 0);

    // Close unnecessary pipe ends in parent
    close(in_pipe[STDIN_FILENO]);
    close(out_pipe[STDOUT_FILENO]);
    close(exec_pipe[STDOUT_FILENO]);

    // Check for exec error
    char buffer[1024];
    if (read(exec_pipe[STDIN_FILENO], buffer, sizeof(errno)) != 0) {
        std::string error_msg = std::strerror(atoi(buffer));
        throw PVCore::PVStreamingCompressorError(
            "Call to compression process failed with the following error message: " + error_msg);
    }
    close(exec_pipe[STDIN_FILENO]);

    _fd = in_pipe[STDOUT_FILENO];
    _status_fd = out_pipe[STDIN_FILENO];
    _child_pid = child_pid;
}

PVCore::PVStreamingCompressor::~PVStreamingCompressor()
{
	if (not _passthrough and not _finished) {
		pvlogger::error()
		    << "PVCore::PVStreamingCompressor::wait_finished() not called before object destruction"
		    << std::endl;
		assert(false);
	}
}

void PVCore::PVStreamingCompressor::write(const std::string& content)
{
	if (_canceled) {
		throw PVStreamingCompressorError("Write attempt to a closed compression stream");
	}

	/*
	 * Avoid potential deadlock when writing content
	 */
	if (not _passthrough and _status_code == 0) {
		int status = 0;
		waitpid(_child_pid, &status, WNOHANG | WUNTRACED);
		if (WIFEXITED(status)) {
			_status_code = WEXITSTATUS(status);
		}
	}

	if (_status_code == 0) {
		if (::write(_fd, content.c_str(), content.size()) < (int)content.size()) {
			std::string error_msg = std::strerror(errno);
			throw PVStreamingCompressorError(
			    std::string("Export failed with the following error message: ") + error_msg);
		}
	}
}

void PVCore::PVStreamingCompressor::do_wait_finished()
{
	if (_canceled) {
		kill(_child_pid, SIGTERM);
	}

	int status = 0;
	pid_t pid = waitpid(_child_pid, &status, 0);

	// throw exception with error message if compression failed
	if (not _canceled and (_status_code != 0 or (pid > 0 && status != 0))) {
		std::string error_msg;
		return_status(&error_msg);

		throw PVStreamingCompressorError(
		    "Compression failed" +
		    (error_msg.empty() ? "" : " with the following error message: " + error_msg));
	}
}

/******************************************************************************
 *
 * PVCore::PVStreamingDecompressor
 *
 ******************************************************************************/

PVCore::PVStreamingDecompressor::PVStreamingDecompressor(const std::string& path)
    : __impl::PVStreamingBase(path)
{
}

PVCore::PVStreamingDecompressor::~PVStreamingDecompressor()
{
	wait_finished();
}

void PVCore::PVStreamingDecompressor::init()
{
    _compressed_chunk_size = 0;

    int input_fd;
    if ((input_fd = open(_path.c_str(), O_RDONLY, 0666)) == -1) {
        throw PVStreamingDecompressorError("Unable to open file '" + _path + "'");
    }

    _init = true;

    if (_passthrough) {
        _fd = input_fd;
        return;
    }

    // Pipes pour les différentes redirections
    int in_pipe[2];
    pipe(in_pipe);

    int out_pipe[2];
    pipe(out_pipe);

    int err_pipe[2];
    pipe(err_pipe);

    int exec_pipe[2];
    pipe(exec_pipe);

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);

    const auto& [args, argv] = executable(_extension, EExecType::DECOMPRESSOR);

    // Redirect stdin depuis in_pipe
    posix_spawn_file_actions_adddup2(&actions, in_pipe[STDIN_FILENO], STDIN_FILENO);

    // Redirect stdout vers out_pipe
    posix_spawn_file_actions_adddup2(&actions, out_pipe[STDOUT_FILENO], STDOUT_FILENO);

    // Redirect stderr vers err_pipe
    posix_spawn_file_actions_adddup2(&actions, err_pipe[STDOUT_FILENO], STDERR_FILENO);

    // Assurez-vous que exec_pipe est fermé à l'exécution pour l'erreur d'execvp
    fcntl(exec_pipe[STDOUT_FILENO], F_SETFD, FD_CLOEXEC);

    pid_t child_pid;
    int status = posix_spawn(&child_pid, /*args[0].c_str()*/"/Users/jib/squey_libs/bin/unpigz", &actions, nullptr, argv.data(), environ);
	if (status != 0) {
        posix_spawn_file_actions_destroy(&actions);
        throw PVStreamingDecompressorError(std::string("Failed to spawn decompression process : ") + strerror(status));
    }

    // Nettoyer les actions de posix_spawn
    posix_spawn_file_actions_destroy(&actions);
    setpgid(child_pid, 0);

    //close(in_pipe[STDIN_FILENO]);
    _write_fd = in_pipe[STDOUT_FILENO];
    //close(err_pipe[STDOUT_FILENO]);
    _status_fd = err_pipe[STDIN_FILENO];
    //close(exec_pipe[STDOUT_FILENO]);

    char buffer[1024];
    if (::read(exec_pipe[STDIN_FILENO], buffer, sizeof(errno)) != 0) {
        int status_code = atoi(buffer);
        if (status_code != 0) {
            std::string error_msg = std::strerror(status_code);
            throw PVStreamingDecompressorError(
                "Call to decompression process failed with the following error message: " +
                error_msg);
        }
    }

    // Thread de transfert des données compressées
    _thread = std::thread([=, this]() {
        static constexpr const size_t buffer_length = 65536;
        std::unique_ptr<char[]> buffer(new char[buffer_length]);

        sigset_t oldset, newset;
        sigemptyset(&newset);
        sigaddset(&newset, SIGPIPE);
        pthread_sigmask(SIG_BLOCK, &newset, &oldset);

        int read_count = 0;
        int write_count = 0;
        char* error_msg;

        while (true) {
            if ((read_count = ::read(input_fd, buffer.get(), buffer_length)) > 0) {
                _compressed_chunk_size += read_count;
                if ((write_count = write(_write_fd, buffer.get(), read_count)) < read_count) {
                    error_msg = std::strerror(errno);
                    break;
                }
            } else {
                error_msg = std::strerror(errno);
                break;
            }
        }

        close(_write_fd);

#if __APPLE__
			sigset_t pending;
			struct timespec ts = {0, 10000000};

			do {
				sigpending(&pending);
				if (sigismember(&pending, SIGPIPE)) {
					struct sigaction sa;
					sa.sa_handler = SIG_DFL;
					sigemptyset(&sa.sa_mask);
					sa.sa_flags = 0;
					sigaction(SIGPIPE, &sa, nullptr);
				}
				nanosleep(&ts, nullptr);
			} while (sigismember(&pending, SIGPIPE));
#else
			siginfo_t si;
			struct timespec ts = {0, 0};
			while (sigtimedwait(&newset, &si, &ts) >= 0 || errno != EAGAIN)
			;
#endif
        pthread_sigmask(SIG_SETMASK, &oldset, nullptr);

        if (read_count < 0) {
            throw PVStreamingDecompressorError(
                "Error while reading compressed file: " + std::string(error_msg));
        }
        if (!(_canceled || _finished) && write_count < read_count) {
            std::string error_msg;
            return_status(&error_msg);

            throw PVStreamingDecompressorError(
                "Error while decompressing file: " + error_msg);
        }
    });

    close(out_pipe[STDOUT_FILENO]);
    _fd = out_pipe[STDIN_FILENO];
}

void PVCore::PVStreamingDecompressor::do_wait_finished()
{
	if (not _init) {
		return;
	}

	close(_write_fd);
	_write_fd = -1;

	kill(_child_pid, SIGTERM);

	_thread.join();

	_init = false;
	_finished = false;
}

PVCore::PVStreamingDecompressor::chunk_sizes_t PVCore::PVStreamingDecompressor::read(char* buffer,
                                                                                     size_t n)
{
	if (not _init or _finished) {
		init();
	}

	if (_canceled) {
		throw PVStreamingDecompressorError("Read attempt from a closed decompression stream");
	}

	int count;
	if ((count = ::read(_fd, buffer, n)) == -1) {
		throw PVStreamingDecompressorError(std::strerror(errno));
	}

	return {count, _passthrough ? count : _compressed_chunk_size.fetch_and(0)};
}

void PVCore::PVStreamingDecompressor::reset()
{
	if (_passthrough) {
		lseek(_fd, 0, SEEK_SET);
	} else {
		if (_init) {
			cancel();
		}
	}
}
