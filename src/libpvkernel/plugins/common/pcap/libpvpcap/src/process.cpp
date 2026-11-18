//
// MIT License
//
// © Squey, 2026
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

#include <libpvpcap/process.h>

#include <algorithm>
#include <cstring>

#ifdef _WIN32
#include <windows.h>

#include <QString>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;
#endif

namespace pvpcap
{

std::string quote_windows_argument(const std::string& argument)
{
	if (not argument.empty() and argument.find_first_of(" \t\n\v\"") == std::string::npos) {
		return argument;
	}
	std::string quoted = "\"";
	size_t backslashes = 0;
	for (const char c : argument) {
		if (c == '\\') {
			backslashes++;
			continue;
		}
		// Backslashes only escape what follows them when a quote does.
		quoted.append(c == '"' ? 2 * backslashes + 1 : backslashes, '\\');
		quoted += c;
		backslashes = 0;
	}
	quoted.append(2 * backslashes, '\\');
	quoted += '"';
	return quoted;
}

#ifdef _WIN32

namespace
{

std::wstring wide(const std::string& s)
{
	return QString::fromStdString(s).toStdWString();
}

std::string last_error()
{
	const DWORD error = GetLastError();
	wchar_t* message = nullptr;
	FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
	                   FORMAT_MESSAGE_IGNORE_INSERTS,
	               nullptr, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
	               reinterpret_cast<LPWSTR>(&message), 0, nullptr);
	std::string text = message != nullptr ? QString::fromWCharArray(message).trimmed().toStdString()
	                                      : "error " + std::to_string(error);
	LocalFree(message);
	return text;
}

HANDLE open_inherited(const std::string& path, bool write)
{
	SECURITY_ATTRIBUTES sa{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
	// Shared for reading: the output of a running child is read as it grows.
	return CreateFileW(wide(path).c_str(), write ? GENERIC_WRITE : GENERIC_READ,
	                   FILE_SHARE_READ | FILE_SHARE_DELETE, &sa,
	                   write ? CREATE_ALWAYS : OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
}

//! The environment of this process, with the given variables added or changed.
std::wstring environment_block(const std::vector<std::pair<std::string, std::string>>& changes)
{
	std::vector<std::wstring> variables;
	if (wchar_t* block = GetEnvironmentStringsW(); block != nullptr) {
		for (const wchar_t* v = block; *v != L'\0'; v += wcslen(v) + 1) {
			variables.emplace_back(v);
		}
		FreeEnvironmentStringsW(block);
	}
	for (const auto& [name, value] : changes) {
		const std::wstring prefix = wide(name) + L"=";
		std::erase_if(variables, [&](const std::wstring& v) {
			return v.size() >= prefix.size() and
			       _wcsnicmp(v.c_str(), prefix.c_str(), prefix.size()) == 0;
		});
		variables.push_back(prefix + wide(value));
	}
	// The block is expected sorted, by name and regardless of case.
	std::sort(variables.begin(), variables.end(), [](const std::wstring& a, const std::wstring& b) {
		return _wcsicmp(a.c_str(), b.c_str()) < 0;
	});
	std::wstring block;
	for (const std::wstring& v : variables) {
		block += v;
		block += L'\0';
	}
	block += L'\0';
	return block;
}

} // namespace

child_process::child_process(const options& opts)
{
	HANDLE input_read = nullptr;
	HANDLE input_write = nullptr;
	if (opts.input.empty()) {
		SECURITY_ATTRIBUTES sa{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
		if (not CreatePipe(&input_read, &input_write, &sa, 1024 * 1024)) {
			throw process_error("Can't create a pipe: " + last_error());
		}
		SetHandleInformation(input_write, HANDLE_FLAG_INHERIT, 0);
	} else {
		input_read = open_inherited(opts.input, false);
		if (input_read == INVALID_HANDLE_VALUE) {
			throw process_error("Can't open '" + opts.input + "': " + last_error());
		}
	}
	const auto close_input_handles = [&]() {
		CloseHandle(input_read);
		if (input_write != nullptr) {
			CloseHandle(input_write);
		}
	};

	HANDLE output = open_inherited(opts.output, true);
	if (output == INVALID_HANDLE_VALUE) {
		const std::string why = last_error();
		close_input_handles();
		throw process_error("Can't create '" + opts.output + "': " + why);
	}
	HANDLE errors = open_inherited(opts.errors, true);
	if (errors == INVALID_HANDLE_VALUE) {
		const std::string why = last_error();
		close_input_handles();
		CloseHandle(output);
		throw process_error("Can't create '" + opts.errors + "': " + why);
	}

	// Only these three handles are inherited: a pipe end another child inherited would keep
	// it open, and the child that reads it would never see its end.
	HANDLE inherited[] = {input_read, output, errors};
	SIZE_T attributes_size = 0;
	InitializeProcThreadAttributeList(nullptr, 1, 0, &attributes_size);
	std::vector<char> attributes(attributes_size);
	auto* attribute_list = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributes.data());
	InitializeProcThreadAttributeList(attribute_list, 1, 0, &attributes_size);
	UpdateProcThreadAttribute(attribute_list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited,
	                          sizeof(inherited), nullptr, nullptr);

	STARTUPINFOEXW si{};
	si.StartupInfo.cb = sizeof(si);
	si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
	si.StartupInfo.hStdInput = input_read;
	si.StartupInfo.hStdOutput = output;
	si.StartupInfo.hStdError = errors;
	si.lpAttributeList = attribute_list;

	std::string command_line;
	for (const std::string& argument : opts.argv) {
		if (not command_line.empty()) {
			command_line += ' ';
		}
		command_line += quote_windows_argument(argument);
	}
	std::wstring wide_command_line = wide(command_line);
	std::wstring environment = environment_block(opts.environment);

	PROCESS_INFORMATION pi{};
	const BOOL started =
	    CreateProcessW(nullptr, wide_command_line.data(), nullptr, nullptr, TRUE,
	                   EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW,
	                   environment.data(), nullptr, &si.StartupInfo, &pi);
	const std::string why = started ? std::string() : last_error();
	DeleteProcThreadAttributeList(attribute_list);
	CloseHandle(input_read);
	CloseHandle(output);
	CloseHandle(errors);
	if (not started) {
		if (input_write != nullptr) {
			CloseHandle(input_write);
		}
		throw process_error("Can't run '" + opts.argv.front() + "': " + why);
	}
	CloseHandle(pi.hThread);
	_process = pi.hProcess;
	_input = input_write;
}

child_process::~child_process()
{
	close_input();
	if (not _waited) {
		kill();
		wait();
	}
	CloseHandle(_process);
}

bool child_process::write(const uint8_t* data, size_t size)
{
	while (size > 0) {
		DWORD written = 0;
		const DWORD chunk = static_cast<DWORD>(std::min<size_t>(size, 1 << 30));
		if (not WriteFile(_input, data, chunk, &written, nullptr)) {
			return false;
		}
		data += written;
		size -= written;
	}
	return true;
}

void child_process::close_input()
{
	if (_input != nullptr) {
		CloseHandle(_input);
		_input = nullptr;
	}
}

child_process::status child_process::wait()
{
	if (not _waited) {
		WaitForSingleObject(_process, INFINITE);
		DWORD code = 0;
		GetExitCodeProcess(_process, &code);
		_status.exited = true;
		_status.code = static_cast<int>(code);
		_waited = true;
	}
	return _status;
}

void child_process::kill()
{
	// The handle keeps the process object: it can't name another process, even ended.
	TerminateProcess(_process, 1);
}

#else

namespace
{

int open_or_throw(const std::string& path, int flags)
{
	const int fd = open(path.c_str(), flags | O_CLOEXEC, 0644);
	if (fd == -1) {
		throw process_error("Can't open '" + path + "': " + std::strerror(errno));
	}
	return fd;
}

} // namespace

child_process::child_process(const options& opts)
{
	// Writing to a child that died raises SIGPIPE, which would kill the whole application:
	// ignored, the failure comes back as EPIPE from write().
	static const bool sigpipe_ignored = []() {
		std::signal(SIGPIPE, SIG_IGN);
		return true;
	}();
	(void)sigpipe_ignored;

	int input_read = -1;
	int input_write = -1;
	if (opts.input.empty()) {
		int fds[2];
#ifdef __APPLE__
		// No pipe2() there: another thread spawning meanwhile could get these, which only
		// matters to the children spawned elsewhere, POSIX_SPAWN_CLOEXEC_DEFAULT keeping
		// them from those spawned here.
		const bool created = pipe(fds) == 0 and fcntl(fds[0], F_SETFD, FD_CLOEXEC) == 0 and
		                     fcntl(fds[1], F_SETFD, FD_CLOEXEC) == 0;
#else
		const bool created = pipe2(fds, O_CLOEXEC) == 0;
#endif
		if (not created) {
			throw process_error(std::string("Can't create a pipe: ") + std::strerror(errno));
		}
		input_read = fds[0];
		input_write = fds[1];
	} else {
		input_read = open_or_throw(opts.input, O_RDONLY);
	}
	int output = -1;
	int errors = -1;
	try {
		output = open_or_throw(opts.output, O_WRONLY | O_CREAT | O_TRUNC);
		errors = open_or_throw(opts.errors, O_WRONLY | O_CREAT | O_TRUNC);
	} catch (...) {
		close(input_read);
		if (input_write != -1) {
			close(input_write);
		}
		if (output != -1) {
			close(output);
		}
		throw;
	}

	// The child only inherits the three descriptors it is given. Not only those this opens:
	// a pipe of another part of the application, whose write end the child kept, would
	// never end for the process reading it -- which a decompressor feeding the capture
	// does.
	posix_spawn_file_actions_t actions;
	posix_spawn_file_actions_init(&actions);
	posix_spawn_file_actions_adddup2(&actions, input_read, STDIN_FILENO);
	posix_spawn_file_actions_adddup2(&actions, output, STDOUT_FILENO);
	posix_spawn_file_actions_adddup2(&actions, errors, STDERR_FILENO);
#ifdef __GLIBC__
	posix_spawn_file_actions_addclosefrom_np(&actions, STDERR_FILENO + 1);
#endif

	// The child starts with the signal dispositions it would have on its own, not with
	// the SIGPIPE this process ignores, nor with what the spawning thread blocks.
	posix_spawnattr_t attributes;
	posix_spawnattr_init(&attributes);
	sigset_t defaults;
	sigemptyset(&defaults);
	sigaddset(&defaults, SIGPIPE);
	posix_spawnattr_setsigdefault(&attributes, &defaults);
	sigset_t none;
	sigemptyset(&none);
	posix_spawnattr_setsigmask(&attributes, &none);
	short flags = POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK;
#ifdef __APPLE__
	flags |= POSIX_SPAWN_CLOEXEC_DEFAULT;
#endif
	posix_spawnattr_setflags(&attributes, flags);

	std::vector<std::string> variables;
	for (char** v = environ; *v != nullptr; v++) {
		variables.emplace_back(*v);
	}
	for (const auto& [name, value] : opts.environment) {
		const std::string prefix = name + "=";
		std::erase_if(variables, [&](const std::string& v) { return v.starts_with(prefix); });
		variables.push_back(prefix + value);
	}
	std::vector<char*> envp;
	for (std::string& v : variables) {
		envp.push_back(v.data());
	}
	envp.push_back(nullptr);

	std::vector<std::string> argv_storage = opts.argv;
	std::vector<char*> argv;
	for (std::string& argument : argv_storage) {
		argv.push_back(argument.data());
	}
	argv.push_back(nullptr);

	const int failed =
	    posix_spawnp(&_pid, argv[0], &actions, &attributes, argv.data(), envp.data());
	posix_spawn_file_actions_destroy(&actions);
	posix_spawnattr_destroy(&attributes);
	close(input_read);
	close(output);
	close(errors);
	if (failed != 0) {
		if (input_write != -1) {
			close(input_write);
		}
		// posix_spawnp() returns the error rather than setting errno.
		throw process_error("Can't run '" + opts.argv.front() + "': " + std::strerror(failed));
	}
	_input = input_write;
}

child_process::~child_process()
{
	close_input();
	if (not _waited) {
		kill();
		wait();
	}
}

bool child_process::write(const uint8_t* data, size_t size)
{
	while (size > 0) {
		const ssize_t written = ::write(_input, data, size);
		if (written < 0) {
			if (errno == EINTR) {
				continue;
			}
			return false;
		}
		data += written;
		size -= written;
	}
	return true;
}

void child_process::close_input()
{
	if (_input != -1) {
		close(_input);
		_input = -1;
	}
}

child_process::status child_process::wait()
{
	if (not _waited) {
		int wstatus = 0;
#ifdef __APPLE__
		// waitid() can't be relied on there to wait without reaping: kill() may race with
		// the reaping, which a process id only reused long after makes harmless.
		while (waitpid(_pid, &wstatus, 0) == -1 and errno == EINTR) {
		}
		const std::lock_guard<std::mutex> reaping(_reaping);
#else
		// Waited for without being reaped first: until it is, below, its process id can't
		// be given to another process, which kill() would then target.
		siginfo_t info{};
		while (waitid(P_PID, _pid, &info, WEXITED | WNOWAIT) == -1 and errno == EINTR) {
		}
		const std::lock_guard<std::mutex> reaping(_reaping);
		while (waitpid(_pid, &wstatus, 0) == -1 and errno == EINTR) {
		}
#endif
		_status.exited = WIFEXITED(wstatus);
		_status.code = WIFEXITED(wstatus) ? WEXITSTATUS(wstatus) : WTERMSIG(wstatus);
		_waited = true;
	}
	return _status;
}

void child_process::kill()
{
	const std::lock_guard<std::mutex> reaping(_reaping);
	if (not _waited) {
		::kill(_pid, SIGKILL);
	}
}

#endif

} // namespace pvpcap
