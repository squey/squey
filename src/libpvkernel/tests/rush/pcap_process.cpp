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

// The tshark processes get their arguments as given, and nothing else of the application.
//
// Windows hands a program one command line, which the arguments have to be quoted into:
// joined with spaces, a display filter or a path with a space in it became several
// arguments. A child that inherits a pipe end another part of the application opened, as
// the decompressor of a compressed capture does, keeps the pipe open: the import of a
// compressed capture waited forever for the end of the decompressor's input.

#include <pvkernel/core/squey_assert.h>

#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap/process.h"
#include "common.h"

#include <QTemporaryDir>

#include <chrono>
#include <iostream>

#ifndef _WIN32
#include <unistd.h>
#endif

int main()
{
	pvtest::init_ctxt();

	std::cout << "quoting for Windows" << std::endl;
	using pvpcap::quote_windows_argument;
	PV_VALID(quote_windows_argument("tshark"), std::string("tshark"));
	PV_VALID(quote_windows_argument(""), std::string("\"\""));
	PV_VALID(quote_windows_argument(R"(C:\Program Files\Wireshark\tshark.exe)"),
	         std::string(R"("C:\Program Files\Wireshark\tshark.exe")"));
	PV_VALID(quote_windows_argument("ip.addr == 10.0.0.1"), std::string("\"ip.addr == 10.0.0.1\""));
	PV_VALID(quote_windows_argument(R"(http.host == "a b")"),
	         std::string(R"("http.host == \"a b\"")"));
	PV_VALID(quote_windows_argument(R"(C:\dir with space\)"),
	         std::string(R"("C:\dir with space\\")"));
	PV_VALID(quote_windows_argument(R"(a\\b)"), std::string(R"(a\\b)"));

#ifndef _WIN32
	std::cout << "nothing inherited" << std::endl;
	QTemporaryDir dir;
	PV_ASSERT_VALID(dir.isValid(), "a temporary directory", "could not be made");

	// Opened without close-on-exec, as a pipe of another part of the application may be.
	int fds[2];
	PV_VALID(pipe(fds), 0);

	pvpcap::child_process::options opts;
	opts.argv = {"sleep", "3"};
	opts.output = dir.filePath("output").toStdString();
	opts.errors = dir.filePath("errors").toStdString();
	pvpcap::child_process child(opts);

	close(fds[1]);
	const auto start = std::chrono::steady_clock::now();
	char c;
	// The end of the pipe comes now, unless the child kept its write end, until it exits.
	PV_VALID(read(fds[0], &c, 1), ssize_t(0));
	const double waited =
	    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	std::cout << "waited " << waited << " s for the end of the pipe" << std::endl;
	PV_ASSERT_VALID(waited < 1, "the end of a pipe the child inherited", waited);
	close(fds[0]);

	child.kill();
	const pvpcap::child_process::status status = child.wait();
	PV_VALID(status.exited, false);
#endif

	return 0;
}
