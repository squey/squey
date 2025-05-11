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
#ifndef _WIN32
#include <sys/wait.h>
#include <spawn.h>
#else
#include <windows.h>
#endif
#include <unistd.h>
#include <pvlogger.h>
#include <signal.h>
#include <stdlib.h>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/detail/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <cerrno>
#include <iostream>
#include <cstring> // for std::strerror
#include <cassert>
#include <algorithm>
#include <memory>
#include <QString>
#include <filesystem>

#include "pvkernel/core/PVOrderedMap.h"

static constexpr int PIPE_READ = 0;
static constexpr int PIPE_WRITE = 1;

#define OUTPUT_FILENAME_PLACEHOLDER "{{filename}}"

extern char **environ;

const PVCore::PVOrderedMap<std::string, std::pair<std::string, std::string>>
    PVCore::__impl::PVStreamingBase::_supported_compressors = {
#ifdef _WIN32
		{"zip", {"7z a dummy.zip -si\"" OUTPUT_FILENAME_PLACEHOLDER "\" -tzip -so -bb0 -bso0 -bse0 -bsp0", "funzip"}},
		{"bz2", {"pbzip2 -z", "pbzip2 -d"}},
#else	
		{"zip", {"7zz a dummy.zip -si" OUTPUT_FILENAME_PLACEHOLDER " -tzip -so -bb0 -bso0 -bse0 -bsp0", "funzip"}},
		{"bz2", {"lbzip2", "lbzip2 -d"}},
#endif
		{"gz", {"pigz -c", "pigz -d -c"}},
		{"xz", {"xz -T0", "xz -d -T0"}},
		{"zst", {"zstd -c", "zstd -d -c"}}};

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

std::tuple<std::vector<std::string>, std::vector<char*>> PVCore::__impl::PVStreamingBase::executable(const std::string& extension, EExecType type, const std::string& output_name)
{
	std::string exec;
	auto it = _supported_compressors.find(extension);
	if (it != _supported_compressors.end()) {
		if (type == EExecType::COMPRESSOR) {
			exec = it->value().first;
			boost::algorithm::replace_all(exec, OUTPUT_FILENAME_PLACEHOLDER, output_name);
		}
		else {
			exec = it->value().second;
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
	if (_status_fd != -1 and status_msg != nullptr) {
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
	int open_flags = O_CREAT | O_WRONLY;
#ifdef _WIN32
	std::wstring wpath = std::filesystem::path(path).wstring();
	_fd = _wopen
#else
	const std::string& wpath = path;
	_fd = open
#endif
	(wpath.c_str(), open_flags, 0666);
	if (_fd == -1) {
		throw PVCore::PVStreamingCompressorError("Unable to create file '" + path + "'");
	}

	if (_passthrough) {
		return;
	}
	_output_fd = _fd;
	std::string output_name = boost::filesystem::path(_path).filename().stem().string();
#if defined(__linux__) || defined(__APPLE__)
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
	std::tie(_args, std::ignore) = executable(_extension, EExecType::COMPRESSOR, output_name);
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
#elifdef _WIN32
	HANDLE in_pipe_read, in_pipe_write;
    HANDLE err_pipe_read, err_pipe_write;
    
    // Create stdin pipe
    SECURITY_ATTRIBUTES sa = {sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    if (not CreatePipe(&in_pipe_read, &in_pipe_write, &sa, 0)) {
        throw PVStreamingCompressorError("Failed to create input pipe");
    }
    SetHandleInformation(in_pipe_write, HANDLE_FLAG_INHERIT, 0);

    // Create stderr pipe
    if (not CreatePipe(&err_pipe_read, &err_pipe_write, &sa, 0)) {
        CloseHandle(in_pipe_read);
        CloseHandle(in_pipe_write);
        throw PVStreamingCompressorError("Failed to create error pipe");
    }
    SetHandleInformation(err_pipe_read, HANDLE_FLAG_INHERIT, 0);

    // Setup process startup attributes
    STARTUPINFO si = {};
    si.cb = sizeof(STARTUPINFOA);
    si.hStdInput = in_pipe_read;
    si.hStdOutput = (HANDLE)_get_osfhandle(_fd);
    si.hStdError = err_pipe_write;
    si.dwFlags |= STARTF_USESTDHANDLES;

    PROCESS_INFORMATION pi = {};

    // Start new process
	std::tie(_args, std::ignore) = executable(_extension, EExecType::COMPRESSOR, output_name);
	_cmdline = QString::fromStdString(boost::algorithm::join(_args, " ")).toStdWString(); // decompressor
    if (not CreateProcess(
		nullptr,
		_cmdline.data(),
		nullptr,
		nullptr,
		TRUE,
		CREATE_UNICODE_ENVIRONMENT | CREATE_NEW_PROCESS_GROUP,
		nullptr,
		nullptr,
		&si,
		&pi)
	) {
        throw PVStreamingCompressorError("Failed to create process");
    }
	_child_pid = pi.hProcess;
	_fd = _open_osfhandle(reinterpret_cast<intptr_t>(in_pipe_write), _O_RDONLY);

    // Close handles
    // CloseHandle(pi.hProcess);
    // CloseHandle(pi.hThread);
    CloseHandle(in_pipe_read);
    CloseHandle(err_pipe_write);
#endif
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
#if defined(__linux__) || defined(__APPLE__)
		int status = 0;
		waitpid(_child_pid, &status, WNOHANG | WUNTRACED);
		if (WIFEXITED(status)) {
			_status_code = WEXITSTATUS(status);
		}
#elifdef _WIN32
		WaitForInputIdle(_child_pid, INFINITE);
#endif
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
#if defined(__linux__) || defined(__APPLE__)
	if (_canceled) {
		kill(_child_pid, SIGTERM);
	}

	int status = 0;
	pid_t pid = waitpid(_child_pid, &status, 0);

	// throw exception with error message if compression failed
	if (not _canceled and (_status_code != 0 or (pid > 0 && status != 0))) {
#elifdef _WIN32
	if (_canceled) {
		if (not TerminateProcess(_child_pid, 1)) {  // 1 = exit code
			throw std::runtime_error("Failed to terminate process");
		}
	}

	WaitForSingleObject(_child_pid, INFINITE);
    DWORD status = 0;
    if (not GetExitCodeProcess(_child_pid, &status)) {
        throw std::runtime_error("Failed to get exit code");
    }
	CloseHandle(_child_pid);
	if (not _canceled and (_status_code != 0 or (status != 0))) {
#endif
		std::string error_msg;
		return_status(&error_msg);

		throw PVStreamingCompressorError(
		    "Compression failed" +
		    (error_msg.empty() ? "" : " with the following error message: " + error_msg));
	}
	else {
		close(_output_fd);
		_output_fd = -1;
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

#ifdef _WIN32
	std::wstring path = std::filesystem::path(_path).wstring();
	if ((input_fd = _wopen(path.c_str(), O_RDONLY | O_BINARY
#else
	if ((input_fd = open(_path.c_str(), O_RDONLY
#endif
	, 0666)) == -1) {
		throw PVStreamingDecompressorError(std::string("Unable to open file '") + _path + "'");
	}

	_init = true;

	if (_passthrough) {
		_fd = input_fd;
		return;
	}
#if defined(__linux__) || defined(__APPLE__)
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
	_fd = out_pipe[PIPE_READ];
#else // _WIN32
	HANDLE in_pipe_read, in_pipe_write;
    HANDLE out_pipe_read, out_pipe_write;
    HANDLE err_pipe_read, err_pipe_write;

    // Create security attributes for inheritable handles
    SECURITY_ATTRIBUTES sa = {sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};

    // Create input pipe (stdin for child)
    if (not CreatePipe(&in_pipe_read, &in_pipe_write, &sa, 0)) {
        throw PVStreamingDecompressorError("Failed to create input pipe");
    }
    SetHandleInformation(in_pipe_write, HANDLE_FLAG_INHERIT, 0);
	_write_fd = _open_osfhandle(reinterpret_cast<intptr_t>(in_pipe_write), _O_RDWR);
    //_write_fd = in_pipe_write; // Used to write to decompressor

    // Create output pipe (stdout from child)
    if (not CreatePipe(&out_pipe_read, &out_pipe_write, &sa, 0)) {
        CloseHandle(in_pipe_read);
        CloseHandle(in_pipe_write);
        throw PVStreamingDecompressorError("Failed to create output pipe");
    }
    SetHandleInformation(out_pipe_read, HANDLE_FLAG_INHERIT, 0);
	_fd = _open_osfhandle(reinterpret_cast<intptr_t>(out_pipe_read), _O_RDONLY);

    // Create error pipe (stderr from child)
    if (not CreatePipe(&err_pipe_read, &err_pipe_write, &sa, 0)) {
        CloseHandle(in_pipe_read);
        CloseHandle(in_pipe_write);
        CloseHandle(out_pipe_read);
        CloseHandle(out_pipe_write);
        throw PVStreamingDecompressorError("Failed to create error pipe");
    }
    SetHandleInformation(err_pipe_read, HANDLE_FLAG_INHERIT, 0);
    _status_fd = _open_osfhandle(reinterpret_cast<intptr_t>(err_pipe_read), _O_RDONLY); // Used to read error messages

    // Set up the process startup information
    STARTUPINFOW si{};
    si.cb = sizeof(STARTUPINFOW);
    si.hStdInput = in_pipe_read;
    si.hStdOutput = out_pipe_write;
    si.hStdError = err_pipe_write;
    si.dwFlags |= STARTF_USESTDHANDLES;

    PROCESS_INFORMATION pi{};

    // Start new process
	auto it = _supported_compressors.find(_extension);
	assert(it != _supported_compressors.end());
	_cmdline = QString::fromStdString(it->value().second).toStdWString(); // decompressor
    if (not CreateProcessW(
        nullptr,
		_cmdline.data(),
		nullptr,
		nullptr,
		TRUE,
        CREATE_UNICODE_ENVIRONMENT | CREATE_NEW_PROCESS_GROUP,
		nullptr,
		nullptr,
		&si,
		&pi)
	) {
        throw PVStreamingDecompressorError("Failed to create decompression process");
    }

    // Close handles
    // CloseHandle(pi.hProcess);
    // CloseHandle(pi.hThread); // FIXME
    CloseHandle(in_pipe_read);
    CloseHandle(out_pipe_write);
    CloseHandle(err_pipe_write);
#endif

	/**
	 * Write compressed file to pipe to store the compressed read bytes count so far
	 * (used to display proper progression during import)
	 */
	_thread = std::thread([=,this]() {
#if defined(__linux__) || defined(__APPLE__)
		/*
		 * ignore "broken pipe" error
		 */
		sigset_t oldset, newset;
 		sigemptyset(&newset);
		sigaddset(&newset, SIGPIPE);
		pthread_sigmask(SIG_BLOCK, &newset, &oldset);
		signal(SIGUSR1, [](int){});
#endif

		const size_t buffer_length = 65336;

		std::unique_ptr<char[]> buffer(new char[buffer_length]);

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

#ifdef __APPLE__
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
#elifdef __linux__
		siginfo_t si;
		struct timespec ts = {0, 0};
		while (sigtimedwait(&newset, &si, &ts) >= 0 || errno != EAGAIN)
		;
#endif
#if defined(__linux__) || defined(__APPLE__)
		pthread_sigmask(SIG_SETMASK, &oldset, nullptr);
#elifdef _WIN32
		// FIXME
#endif

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

#if defined(__linux__) || defined(__APPLE__)
    kill(_child_pid, SIGTERM);
	pthread_kill(_thread.native_handle(), SIGUSR1);
#elifdef _WIN32
	CloseHandle(_child_pid);
#endif

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