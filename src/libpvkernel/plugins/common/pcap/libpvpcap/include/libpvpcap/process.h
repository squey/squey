/* * MIT License
 *
 * © Squey, 2026
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef LIBPVPCAP_PROCESS_H
#define LIBPVPCAP_PROCESS_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pvpcap
{

struct process_error : public std::runtime_error {
	using std::runtime_error::runtime_error;
};

/**
 * A child process, whose standard output and error go to files.
 *
 * Its standard input is either a file, or a pipe that write() feeds. The child inherits
 * no other file descriptor or handle.
 *
 * Thread-safety: write() and close_input() are for one thread; kill() may be called from
 * any thread at any time, wait() from one thread.
 */
class child_process
{
  public:
	struct options {
		std::vector<std::string> argv; //!< in UTF-8, the program first
		std::string input;             //!< file to read from, a pipe if empty
		std::string output;            //!< file to write the standard output to
		std::string errors;            //!< file to write the standard error to
		std::vector<std::pair<std::string, std::string>> environment; //!< what to add or change
	};

	struct status {
		bool exited = false; //!< false if a signal, or a kill(), ended it
		int code = 0;        //!< exit code, or signal number
	};

	/**
	 * Starts the process.
	 *
	 * @throw process_error if it could not be started, with the reason why.
	 */
	explicit child_process(const options& opts);

	//! Kills the process if it still runs, and waits for it.
	~child_process();

	child_process(const child_process&) = delete;
	child_process& operator=(const child_process&) = delete;

	/**
	 * Writes to the standard input of the process, blocking until all of it is written.
	 *
	 * @return false if the process no longer reads it
	 */
	bool write(const uint8_t* data, size_t size);

	//! Closes the standard input, which tells the process there is nothing more to read.
	void close_input();

	//! Waits for the process to end.
	status wait();

	void kill();

  private:
#ifdef _WIN32
	void* _process = nullptr;
	void* _input = nullptr;
#else
	int _pid = -1;
	int _input = -1;
#endif
	std::mutex _reaping; //!< so that kill() never targets a process id reaped meanwhile
	std::atomic<bool> _waited = false;
	status _status;
};

/**
 * Quotes an argument the way the C runtime of Windows splits a command line.
 */
std::string quote_windows_argument(const std::string& argument);

} // namespace pvpcap

#endif // LIBPVPCAP_PROCESS_H
