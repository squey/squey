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
#include <spawn.h>

#include "pvkernel/core/PVOrderedMap.h"

static constexpr int PIPE_READ = 0;
static constexpr int PIPE_WRITE = 1;

extern char **environ;

const PVCore::PVOrderedMap<std::string, std::pair<std::string, std::string>>
    PVCore::__impl::PVStreamingBase::_supported_compressors = {
        {"gz", {"pigz", "unpigz"}}, {"bz2", {"lbzip2", "lbunzip2"}}, {"zip", {"zip", "funzip"}}, {"xz", {"xz -T0", "unxz -T0"}}};

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

PVCore::PVStreamingCompressor::PVStreamingCompressor(const std::string& path)
    : __impl::PVStreamingBase(path)
{
	_fd = open(path.c_str(), O_CREAT | O_WRONLY, 0666);
	if (_fd == -1) {
		throw PVCore::PVStreamingCompressorError("Unable to create file '" + path + "'");
	}

	if (_passthrough) {
		return;
	}

	posix_spawn_file_actions_t actions;
	posix_spawn_file_actions_init(&actions);

	// Used to forward data to compressor
	int in_pipe[2];
	pipe(in_pipe);
	posix_spawn_file_actions_adddup2(&actions, in_pipe[PIPE_READ], STDIN_FILENO);
	posix_spawn_file_actions_addclose(&actions, in_pipe[PIPE_WRITE]);

	// redirect std::out to file
	posix_spawn_file_actions_adddup2(&actions, _fd, STDOUT_FILENO);

	// Used to get error message back from compressor
	int err_pipe[2];
	pipe(err_pipe);
	_status_fd = err_pipe[PIPE_READ];
	posix_spawn_file_actions_adddup2(&actions, err_pipe[PIPE_WRITE], STDERR_FILENO);
	posix_spawn_file_actions_addclose(&actions, err_pipe[PIPE_READ]);

	// Spawn new process
	std::tie(_args, std::ignore) = executable(_extension, EExecType::COMPRESSOR);
	_argv.clear();
	for (const auto& arg : _args) {
		_argv.push_back((char*)arg.data());
	}
	_argv.push_back(nullptr);
	int status_code = posix_spawnp(&_child_pid, _argv[0], &actions, nullptr, _argv.data(), environ);
	posix_spawn_file_actions_destroy(&actions);

	if (status_code != 0) {
		std::string error_msg = std::strerror(status_code);
		throw PVStreamingCompressorError(
			"Call to compression process failed with the following error message: " +
			error_msg);
	}

	_fd = in_pipe[PIPE_WRITE];
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
		throw PVStreamingDecompressorError(std::string("Unable to open file '") + _path + "'");
	}

	_init = true;

	if (_passthrough) {
		_fd = input_fd;
		return;
	}

	posix_spawn_file_actions_t actions;
	posix_spawn_file_actions_init(&actions);

	// Used to forward data to decompressor
	int in_pipe[2];
	pipe(in_pipe);
	_write_fd = in_pipe[PIPE_WRITE];
	posix_spawn_file_actions_adddup2(&actions, in_pipe[PIPE_READ], STDIN_FILENO);
	posix_spawn_file_actions_addclose(&actions, in_pipe[PIPE_WRITE]);

	// Used to get decompressed data
	int out_pipe[2];
	pipe(out_pipe);
	posix_spawn_file_actions_adddup2(&actions, out_pipe[PIPE_WRITE], STDOUT_FILENO);
	posix_spawn_file_actions_addclose(&actions, out_pipe[PIPE_READ]);

	// Used to get error back from decompressor
	int err_pipe[2];
	pipe(err_pipe);
	_status_fd = err_pipe[PIPE_READ];
	posix_spawn_file_actions_adddup2(&actions, err_pipe[PIPE_WRITE], STDERR_FILENO);
	posix_spawn_file_actions_addclose(&actions, err_pipe[PIPE_READ]);

	// Spawn new process
	std::tie(_args, std::ignore) = executable(_extension, EExecType::DECOMPRESSOR);
	_argv.clear();
	for (const auto& arg : _args) {
		_argv.push_back((char*)arg.data());
	}
	_argv.push_back(nullptr);
	int status_code = posix_spawnp(&_child_pid, _argv[0], &actions, nullptr, _argv.data(), environ);
	posix_spawn_file_actions_destroy(&actions);

	if (status_code != 0) {
		std::string error_msg = std::strerror(status_code);
		throw PVStreamingDecompressorError(
			"Call to decompression process failed with the following error message: " +
			error_msg);
	}

	setpgid(_child_pid, 0);
	close(out_pipe[PIPE_WRITE]);

	/**
	 * Write compressed file to pipe to store the compressed read bytes count so far
	 * (used to display proper progression during import)
	 */
	_thread = std::thread([=,this]() {
		signal(SIGUSR1, [](int){});
		const size_t buffer_length = 65336;

		std::unique_ptr<char[]> buffer(new char[buffer_length]);

		/*
		 * ignore "broken pipe" error
		 */
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
				std::string("Error while reading compressed file :") + error_msg);
		}
		if ((not _canceled and not _finished) and write_count < read_count) {
			std::string error_msg;
			return_status(&error_msg);

			throw PVStreamingDecompressorError(
				std::string("Error while decompressing file : ") + error_msg);
		}
	});

	_fd = out_pipe[PIPE_READ];
}

void PVCore::PVStreamingDecompressor::do_wait_finished()
{
	if (not _init) {
		return;
	}

#ifdef __linux__
	close(_write_fd);
	_write_fd = -1;
#endif

    kill(_child_pid, SIGTERM);

	pthread_kill(_thread.native_handle(), SIGUSR1);

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