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
#include <csignal>
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
#elif __APPLE__
		// bsdtar properly handles the data descriptors used by streamed zip archives,
		// unlike funzip which exits with an error on them
		{"zip", {"7zz a dummy.zip -si" OUTPUT_FILENAME_PLACEHOLDER " -tzip -so -bb0 -bso0 -bse0 -bsp0", "bsdtar -xOf -"}},
		{"bz2", {"lbzip2", "lbzip2 -d"}},
#else // __linux__
        {"zip", {"7zzs a dummy.zip -si" OUTPUT_FILENAME_PLACEHOLDER " -tzip -so -bb0 -bso0 -bse0 -bsp0", "bsdtar -xOf -"}},
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

static std::string extension_from_path(const std::string& path)
{
	const size_t dot_pos = path.rfind('.');
	if (dot_pos == std::string::npos) {
		return {};
	}
	return boost::algorithm::to_lower_copy(path.substr(dot_pos + 1));
}

PVCore::__impl::PVStreamingBase::PVStreamingBase(const std::string& path)
    : _path(path)
    , _extension(extension_from_path(path))
    , _passthrough(_supported_compressors.find(_extension) == _supported_compressors.end())
{
#if defined(__linux__) || defined(__APPLE__)
	// Feeding a child process that has stopped reading raises SIGPIPE, and the two
	// write paths below block it per thread so the failure surfaces as EPIPE, which
	// they already handle. That is not enough: the signal was seen killing the main
	// thread while it waited in read() for the feeder to finish, since only the
	// feeder had it masked. Ignoring it once for the process is the usual answer for
	// code that writes to pipes, and leaves the EPIPE handling in charge.
	static const bool sigpipe_ignored = []() {
		std::signal(SIGPIPE, SIG_IGN);
		return true;
	}();
	(void)sigpipe_ignored;
#endif
}

PVCore::__impl::PVStreamingBase::~PVStreamingBase()
{
	if (_status_fd != -1) {
		close(_status_fd);
		_status_fd = -1;
	}
	// in case the constructor threw or wait_finished() was never called
	if (_fd != -1) {
		close(_fd);
		_fd = -1;
	}
	if (_output_fd != -1) {
		close(_output_fd);
		_output_fd = -1;
	}
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

std::vector<std::string> PVCore::__impl::PVStreamingBase::executable(const std::string& extension, EExecType type, const std::string& output_name)
{
	std::string exec;
	auto it = _supported_compressors.find(extension);
	if (it != _supported_compressors.end()) {
		exec = type == EExecType::COMPRESSOR ? it->value().first : it->value().second;
	}
	if (exec.empty()) {
		return {};
	}

	std::vector<std::string> args;
	boost::algorithm::split(args, exec, boost::is_any_of(" "));

	// substitute the placeholder after splitting so that output names containing
	// spaces remain a single argument
	for (std::string& arg : args) {
		boost::algorithm::replace_all(arg, OUTPUT_FILENAME_PLACEHOLDER, output_name);
	}

	return args;
}

#if defined(__linux__) || defined(__APPLE__)
void PVCore::__impl::PVStreamingBase::build_argv()
{
	_argv.clear();
	for (std::string& arg : _args) {
		_argv.push_back(arg.data());
	}
	_argv.push_back(nullptr);
}
#endif

int PVCore::__impl::PVStreamingBase::return_status(std::string* status_msg /* = nullptr */)
{
	if (_status_fd != -1 and status_msg != nullptr) {
		char buffer[1024];
		while (true) {
			ssize_t read_count = ::read(_status_fd, buffer, sizeof(buffer));
			if (read_count < 0 and errno == EINTR) {
				continue;
			}
			if (read_count <= 0) {
				break;
			}
			_status_msg.append(buffer, read_count);
		}
		close(_status_fd);
		_status_fd = -1;
		boost::algorithm::trim(_status_msg);
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
	// O_TRUNC: exporting over an existing longer file must not leave stale content behind
	int open_flags = O_CREAT | O_WRONLY | O_TRUNC;
#ifdef _WIN32
	// without O_BINARY the CRT translates every '\n' into "\r\n", which would make
	// uncompressed exports differ from compressed ones (written by the child process)
	open_flags |= O_BINARY;
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
	if (pipe(in_pipe) != 0) {
		posix_spawn_file_actions_destroy(&actions);
		throw PVStreamingCompressorError("Unable to create compression pipe");
	}
	posix_spawn_file_actions_adddup2(&actions, in_pipe[PIPE_READ], STDIN_FILENO);
	posix_spawn_file_actions_addclose(&actions, in_pipe[PIPE_WRITE]);

	// redirect std::out to file
	posix_spawn_file_actions_adddup2(&actions, _fd, STDOUT_FILENO);

	// Used to get error message back from compressor
	int err_pipe[2];
	if (pipe(err_pipe) != 0) {
		posix_spawn_file_actions_destroy(&actions);
		close(in_pipe[PIPE_READ]);
		close(in_pipe[PIPE_WRITE]);
		throw PVStreamingCompressorError("Unable to create compression pipe");
	}
	_status_fd = err_pipe[PIPE_READ];
	posix_spawn_file_actions_adddup2(&actions, err_pipe[PIPE_WRITE], STDERR_FILENO);
	posix_spawn_file_actions_addclose(&actions, err_pipe[PIPE_READ]);

	// Spawn new process
	_args = executable(_extension, EExecType::COMPRESSOR, output_name);
	build_argv();
	int status_code = posix_spawnp(&_child_pid, _argv[0], &actions, nullptr, _argv.data(), environ);
	posix_spawn_file_actions_destroy(&actions);

	// close the child's ends of the pipes in the parent process: keeping the read end
	// of the input pipe open would prevent EPIPE detection on child death, and keeping
	// the write end of the error pipe open would make return_status() block forever
	// waiting for EOF
	close(in_pipe[PIPE_READ]);
	close(err_pipe[PIPE_WRITE]);

	if (status_code != 0) {
		close(in_pipe[PIPE_WRITE]);
		_output_fd = -1; // still referenced by _fd, closed by the base destructor
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
	_args = executable(_extension, EExecType::COMPRESSOR, output_name);
	_cmdline = QString::fromStdString(boost::algorithm::join(_args, " ")).toStdWString();
    if (not CreateProcessW(
		nullptr,
		_cmdline.data(),
		nullptr,
		nullptr,
		TRUE,
		CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW,
		nullptr,
		nullptr,
		&si,
		&pi)
	) {
        throw PVStreamingCompressorError("Failed to create process");
    }
	_child_pid = pi.hProcess; // closed by do_wait_finished()
	_fd = _open_osfhandle(reinterpret_cast<intptr_t>(in_pipe_write), _O_RDONLY);

    // Close handles
    CloseHandle(pi.hThread);
    CloseHandle(in_pipe_read);
    CloseHandle(err_pipe_write);
#endif
}

PVCore::PVStreamingCompressor::~PVStreamingCompressor()
{
	if (_finished) {
		return;
	}

	if (not _passthrough and std::uncaught_exceptions() == 0) {
		pvlogger::error()
		    << "PVCore::PVStreamingCompressor::wait_finished() not called before object destruction"
		    << std::endl;
		assert(false);
	}

	// cleanup anyway to avoid leaking the child process and file descriptors
	try {
		cancel();
	} catch (...) {
	}
}

void PVCore::PVStreamingCompressor::write(const std::string& content)
{
	if (_canceled) {
		throw PVStreamingCompressorError("Write attempt to a closed compression stream");
	}

	if (not _passthrough and _status_code == 0) {
#if defined(__linux__) || defined(__APPLE__)
		// check if the compression process died prematurely
		int status = 0;
		if (waitpid(_child_pid, &status, WNOHANG | WUNTRACED) == _child_pid and
		    WIFEXITED(status)) {
			_status_code = WEXITSTATUS(status);
		}
#endif
	}

	if (_status_code != 0) {
		return; // error reported by wait_finished()
	}

#if defined(__linux__) || defined(__APPLE__)
	// block SIGPIPE so that a dead compression process results in EPIPE instead of
	// killing the calling thread
	sigset_t sigpipe_set, old_set;
	sigemptyset(&sigpipe_set);
	sigaddset(&sigpipe_set, SIGPIPE);
	pthread_sigmask(SIG_BLOCK, &sigpipe_set, &old_set);
#endif

	size_t written = 0;
	int write_errno = 0;
	while (written < content.size()) {
		ssize_t count = ::write(_fd, content.data() + written, content.size() - written);
		if (count >= 0) {
			written += count;
			continue;
		}
		if (errno == EINTR) {
			continue;
		}
		write_errno = errno;
		break;
	}

#if defined(__linux__) || defined(__APPLE__)
	if (write_errno == EPIPE) {
		// consume the pending SIGPIPE before restoring the signal mask
		sigset_t pending;
		sigpending(&pending);
		if (sigismember(&pending, SIGPIPE)) {
			int sig;
			sigwait(&sigpipe_set, &sig);
		}
	}
	pthread_sigmask(SIG_SETMASK, &old_set, nullptr);
#endif

	if (written < content.size()) {
		std::string error_msg;
		if (write_errno == EPIPE) {
			// the compression process died: report its error output if any
			return_status(&error_msg);
		}
		if (error_msg.empty()) {
			error_msg = std::strerror(write_errno);
		}
		throw PVStreamingCompressorError(
		    std::string("Export failed with the following error message: ") + error_msg);
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

	close(_output_fd);
	_output_fd = -1;

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
	close(_output_fd);
	_output_fd = -1;
	if (not _canceled and (_status_code != 0 or (status != 0))) {
#endif
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

#ifdef _WIN32
	std::wstring path = std::filesystem::path(_path).wstring();
	if ((input_fd = _wopen(path.c_str(), O_RDONLY | O_BINARY
#else
	if ((input_fd = open(_path.c_str(), O_RDONLY
#endif
	, 0666)) == -1) {
		throw PVStreamingDecompressorError(std::string("Unable to open file '") + _path + "'");
	}

	if (_passthrough) {
		_fd = input_fd;
		_init = true;
		return;
	}

	{
		std::lock_guard<std::mutex> lock(_thread_error_mutex);
		_thread_error.clear();
	}

#if defined(__linux__) || defined(__APPLE__)
	posix_spawn_file_actions_t actions;
	posix_spawn_file_actions_init(&actions);

	// Used to forward data to decompressor
	int in_pipe[2];
	// Used to get decompressed data
	int out_pipe[2];
	// Used to get error back from decompressor
	int err_pipe[2];
	if (pipe(in_pipe) != 0 or pipe(out_pipe) != 0 or pipe(err_pipe) != 0) {
		posix_spawn_file_actions_destroy(&actions);
		close(input_fd);
		throw PVStreamingDecompressorError("Unable to create decompression pipe");
	}
	_write_fd = in_pipe[PIPE_WRITE];
	posix_spawn_file_actions_adddup2(&actions, in_pipe[PIPE_READ], STDIN_FILENO);
	posix_spawn_file_actions_addclose(&actions, in_pipe[PIPE_WRITE]);

	posix_spawn_file_actions_adddup2(&actions, out_pipe[PIPE_WRITE], STDOUT_FILENO);
	posix_spawn_file_actions_addclose(&actions, out_pipe[PIPE_READ]);

	_status_fd = err_pipe[PIPE_READ];
	posix_spawn_file_actions_adddup2(&actions, err_pipe[PIPE_WRITE], STDERR_FILENO);
	posix_spawn_file_actions_addclose(&actions, err_pipe[PIPE_READ]);

	// Spawn new process
	_args = executable(_extension, EExecType::DECOMPRESSOR);
	build_argv();
	int status_code = posix_spawnp(&_child_pid, _argv[0], &actions, nullptr, _argv.data(), environ);
	posix_spawn_file_actions_destroy(&actions);

	// close the child's ends of the pipes in the parent process: keeping the read end
	// of the input pipe open would prevent the feeder thread from getting EPIPE when
	// the child dies, and keeping the write end of the error pipe open would make
	// return_status() block forever waiting for EOF
	close(in_pipe[PIPE_READ]);
	close(out_pipe[PIPE_WRITE]);
	close(err_pipe[PIPE_WRITE]);

	if (status_code != 0) {
		close(input_fd);
		close(in_pipe[PIPE_WRITE]);
		close(out_pipe[PIPE_READ]);
		_write_fd = -1;
		std::string error_msg = std::strerror(status_code);
		throw PVStreamingDecompressorError(
			"Call to decompression process failed with the following error message: " +
			error_msg);
	}

	setpgid(_child_pid, 0);
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
        CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW,
		nullptr,
		nullptr,
		&si,
		&pi)
	) {
        throw PVStreamingDecompressorError("Failed to create decompression process");
    }
	_child_pid = pi.hProcess; // closed by do_wait_finished()

    // Close handles
    CloseHandle(pi.hThread);
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
		// use sigaction() without SA_RESTART so that SIGUSR1 actually interrupts
		// blocking syscalls (signal() implies SA_RESTART with glibc)
		struct sigaction usr1_action {};
		usr1_action.sa_handler = [](int) {};
		sigemptyset(&usr1_action.sa_mask);
		sigaction(SIGUSR1, &usr1_action, nullptr);
#endif

		const size_t buffer_length = 65536;

		std::unique_ptr<char[]> buffer(new char[buffer_length]);

		ssize_t read_count = 0;
		bool write_failed = false;
		int write_errno = 0;
		std::string error_msg;

		while (true) {
			read_count = ::read(input_fd, buffer.get(), buffer_length);
			if (read_count < 0 and errno == EINTR and not _canceled and not _finished) {
				continue;
			}
			if (read_count <= 0) {
				if (read_count < 0) {
					error_msg = std::strerror(errno);
				}
				break;
			}
			_compressed_chunk_size += read_count;

			ssize_t written = 0;
			while (written < read_count) {
				ssize_t write_count = write(_write_fd, buffer.get() + written, read_count - written);
				if (write_count >= 0) {
					written += write_count;
					continue;
				}
				if (errno == EINTR and not _canceled and not _finished) {
					continue;
				}
				write_errno = errno;
				error_msg = std::strerror(errno);
				write_failed = true;
				break;
			}
			if (write_failed) {
				break;
			}
		}

		close(_write_fd);
		close(input_fd);

#ifdef __APPLE__
		// Consume the pending SIGPIPE, exactly as PVStreamingCompressor::write() does.
		// Restoring SIG_DFL here, as this used to, did the opposite of draining it: the
		// signal stayed pending with its terminating disposition and killed the process
		// as soon as the mask was lifted below -- while a broken pipe is a normal
		// outcome here, a decompressor being allowed to stop reading. sigwait() would
		// block if nothing were pending, hence the sigpending() guard around it.
		sigset_t pending;
		sigpending(&pending);
		while (sigismember(&pending, SIGPIPE)) {
			int sig;
			sigwait(&newset, &sig);
			sigpending(&pending);
		}
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

		// report errors through read() as throwing from a thread would call std::terminate().
		// a broken pipe is not reported here: a decompressor is allowed to stop reading
		// once it has produced its whole output, so only its exit code can tell whether
		// it actually failed. read() checks it once this thread is joined.
		if (not _canceled and not _finished) {
			if (read_count < 0) {
				std::lock_guard<std::mutex> lock(_thread_error_mutex);
				_thread_error = std::string("Error while reading compressed file: ") + error_msg;
			} else if (write_failed and write_errno != EPIPE) {
				std::lock_guard<std::mutex> lock(_thread_error_mutex);
				_thread_error = "Error while feeding the decompression process: " + error_msg;
			}
		}
	});

	_init = true;
}

void PVCore::PVStreamingDecompressor::do_wait_finished()
{
	if (not _init) {
		return;
	}

#if defined(__linux__) || defined(__APPLE__)
	if (_child_pid > 0) {
		kill(_child_pid, SIGTERM);
	}
	if (_thread.joinable()) {
		pthread_kill(_thread.native_handle(), SIGUSR1);
	}
#elifdef _WIN32
	// terminate the child so that the feeder thread gets a broken pipe instead of
	// staying blocked on write() forever
	if (_child_pid != INVALID_HANDLE_VALUE) {
		TerminateProcess(_child_pid, 1);
		WaitForSingleObject(_child_pid, INFINITE);
		CloseHandle(_child_pid);
		_child_pid = INVALID_HANDLE_VALUE;
	}
#endif

	if (_thread.joinable()) {
		_thread.join();
	}
	// _write_fd and input_fd are closed by the feeder thread
	_write_fd = -1;

#if defined(__linux__) || defined(__APPLE__)
	// reap the child process to avoid leaving zombies behind
	if (_child_pid > 0) {
		waitpid(_child_pid, nullptr, 0);
		_child_pid = -1;
	}
#endif

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

	ssize_t count;
	while ((count = ::read(_fd, buffer, n)) == -1 and errno == EINTR) {
	}
	if (count == -1) {
		throw PVStreamingDecompressorError(std::strerror(errno));
	}

	if (count == 0 and not _passthrough) {
		// EOF: make sure the feeder thread is done, then check how the decompression
		// process ended. A decompressor fed with corrupted data exits with an error
		// without necessarily breaking any pipe, so its exit code is the only reliable
		// failure indicator.
		[[maybe_unused]] bool killed = false;
#if defined(__linux__) || defined(__APPLE__)
		if (_child_pid > 0) {
			kill(_child_pid, SIGTERM); // unblock the feeder thread if the child hangs
		}
#elifdef _WIN32
		// the child closed its output: let it exit on its own, and terminate it if it
		// does not, so that the feeder thread cannot stay blocked on write()
		if (_child_pid != INVALID_HANDLE_VALUE and
		    WaitForSingleObject(_child_pid, 5000) != WAIT_OBJECT_0) {
			TerminateProcess(_child_pid, 1);
			killed = true;
		}
#endif
		// join before reading the exit status: the feeder thread also uses _status_fd
		if (_thread.joinable()) {
			_thread.join();
		}

		bool child_failed = false;
#if defined(__linux__) || defined(__APPLE__)
		if (_child_pid > 0) {
			int status = 0;
			if (waitpid(_child_pid, &status, 0) == _child_pid) {
				_child_pid = -1;
				// a process killed by the SIGTERM above did not fail by itself
				child_failed = (WIFEXITED(status) and WEXITSTATUS(status) != 0) or
				               (WIFSIGNALED(status) and WTERMSIG(status) != SIGTERM);
			}
		}
#elifdef _WIN32
		// funzip exits with an error on the archives produced by "7z -si" (which use
		// data descriptors) even though the extracted data is correct, so its exit
		// code cannot be used to detect a failure
		DWORD exit_code = 0;
		if (not killed and _extension != "zip" and _child_pid != INVALID_HANDLE_VALUE and
		    GetExitCodeProcess(_child_pid, &exit_code)) {
			child_failed = exit_code != 0;
		}
#endif
		std::lock_guard<std::mutex> lock(_thread_error_mutex);
		if (child_failed and _thread_error.empty()) {
			std::string status_msg;
			return_status(&status_msg);
			_thread_error =
			    "Error while decompressing file" + (status_msg.empty() ? "" : ": " + status_msg);
		}
		if (not _thread_error.empty()) {
			throw PVStreamingDecompressorError(_thread_error);
		}
	}

	return {static_cast<size_t>(count),
	        _passthrough ? static_cast<size_t>(count) : _compressed_chunk_size.exchange(0)};
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
