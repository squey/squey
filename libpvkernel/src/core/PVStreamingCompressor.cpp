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

#include <cerrno>
#include <fstream>
#include <iostream>
#include <cstring> // for std::strerror

#include <cassert>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <boost/algorithm/string.hpp>

#include <pvlogger.h>

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

	// Used to check if execlp failed
	int exec_pipe[2];
	pipe(exec_pipe);

	const auto & [args, argv] = executable(_extension, EExecType::COMPRESSOR);
	switch (_child_pid = vfork()) {
	case 0: { // child process
		setpgid(0, 0);

		// redirect parent pipe to std::in
		close(in_pipe[STDOUT_FILENO]);
		dup2(in_pipe[STDIN_FILENO], STDIN_FILENO);

		// redirect std::err to parent
		close(out_pipe[STDIN_FILENO]);
		dup2(out_pipe[STDOUT_FILENO], STDERR_FILENO);

		// set close-on-exec flag to check execlp status
		close(exec_pipe[STDIN_FILENO]);
		fcntl(exec_pipe[STDOUT_FILENO], F_SETFD, FD_CLOEXEC);

		// redirect std::out to file
		int compression_fd = open(path.c_str(), O_CREAT | O_WRONLY | O_CLOEXEC, 0666);
		if (compression_fd == -1) {
			std::cerr << "Unable to create file '" + path + "'" << std::endl;
			close(in_pipe[STDIN_FILENO]);
			exit(-1);
		}
		dup2(compression_fd, STDOUT_FILENO);

		if (execvp(args[0].c_str(), argv.data())) {
			_exit(-1);
		}
	} break;
	default: { // parent process
		setpgid(_child_pid, 0);

		close(in_pipe[STDIN_FILENO]);

		close(out_pipe[STDOUT_FILENO]);
		_status_fd = out_pipe[STDIN_FILENO];

		close(exec_pipe[STDOUT_FILENO]);
		char buffer[1024];
		if (read(exec_pipe[STDIN_FILENO], buffer, sizeof(errno)) != 0) {
			std::string error_msg = std::strerror(atoi(buffer));
			throw PVStreamingCompressorError(
			    "Call to compression process failed with the following error message: " +
			    error_msg);
		}
	}
	}

	_fd = in_pipe[STDOUT_FILENO];
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

	// Used to forward data to decompressor
	int in_pipe[2];
	pipe(in_pipe);

	// Used to get decompressed data
	int out_pipe[2];
	pipe(out_pipe);

	// Used to get error back from decompressor
	int err_pipe[2];
	pipe(err_pipe);

	// Used to check if execlp failed
	int exec_pipe[2];
	pipe(exec_pipe);

	/**
	 * We need to spawn a new process to have unshared file descriptors
	 */
	const auto & [args, argv] = executable(_extension, EExecType::DECOMPRESSOR);
	switch (_child_pid = vfork()) {
	case 0: { // child process
		setpgid(0, 0);

		// redirect parent pipe to std::in
		close(in_pipe[STDOUT_FILENO]);
		dup2(in_pipe[STDIN_FILENO], STDIN_FILENO);

		// redirect std::out to parent pipe
		close(out_pipe[STDIN_FILENO]);
		dup2(out_pipe[STDOUT_FILENO], STDOUT_FILENO);

		// redirect std::err to parent
		close(err_pipe[STDIN_FILENO]);
		dup2(err_pipe[STDOUT_FILENO], STDERR_FILENO);

		// set close-on-exec flag to check execvp status
		close(exec_pipe[STDIN_FILENO]);
		fcntl(exec_pipe[STDOUT_FILENO], F_SETFD, FD_CLOEXEC);

		if (execvp(args[0].c_str(), argv.data()) == -1) {
			write(exec_pipe[STDOUT_FILENO], std::to_string(errno).c_str(), sizeof(errno));
			_exit(errno);
		}
	} break;
	default: { // parent process
		setpgid(_child_pid, 0);

		close(in_pipe[STDIN_FILENO]);
		_write_fd = in_pipe[STDOUT_FILENO];

		close(err_pipe[STDOUT_FILENO]);
		_status_fd = err_pipe[STDIN_FILENO];

		close(exec_pipe[STDOUT_FILENO]);
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

		/**
		 * Write compressed file to pipe to store the compressed read bytes count so far
		 * (used to display proper progression during import)
		 */
		_thread = std::thread([=,this]() {
			static constexpr const size_t buffer_length = 65536;
			std::unique_ptr<char[]> buffer(new char[buffer_length]);

			/*
			 * ignore "broken pipe" error
			 */
			sigset_t oldset, newset;
			siginfo_t si;
			struct timespec ts = {0, 0};
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

			// reset default SIGPIPE handler
			while (sigtimedwait(&newset, &si, &ts) >= 0 || errno != EAGAIN)
				;
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

		close(out_pipe[STDOUT_FILENO]);
		_fd = out_pipe[STDIN_FILENO];
	}
	}
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