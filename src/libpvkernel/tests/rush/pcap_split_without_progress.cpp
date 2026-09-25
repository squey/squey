//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

// A capture split with no progress callback is split.
//
// The split reports its progress once past a megabyte of packets, through a lambda
// that called the caller's callback whether or not there was one: past that
// megabyte, a caller passing none got std::bad_function_call. The captures the
// other tests read are far smaller, so this one writes three megabytes of its own.

#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNrawCacheManager.h>

#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap/pcap_splitter.h"

#include <QTemporaryDir>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "common.h"

namespace
{

template <typename T>
void put(std::ofstream& out, T value)
{
	out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

//! A classic pcap file, in the byte order of this machine, of Ethernet frames of zeros.
void write_capture(const std::string& path, uint32_t packets, uint32_t packet_size)
{
	std::ofstream out(path, std::ios::binary);
	put<uint32_t>(out, 0xa1b2c3d4); // magic, microsecond timestamps
	put<uint16_t>(out, 2);          // version major
	put<uint16_t>(out, 4);          // version minor
	put<int32_t>(out, 0);           // this zone
	put<uint32_t>(out, 0);          // timestamp accuracy
	put<uint32_t>(out, 65535);      // snapshot length
	put<uint32_t>(out, 1);          // Ethernet
	const std::vector<char> frame(packet_size, 0);
	for (uint32_t i = 0; i < packets; ++i) {
		put<uint32_t>(out, 1000000000 + i); // seconds
		put<uint32_t>(out, 0);              // microseconds
		put<uint32_t>(out, packet_size);    // captured
		put<uint32_t>(out, packet_size);    // on the wire
		out.write(frame.data(), std::streamsize(frame.size()));
	}
}

} // namespace

int main()
{
	pvtest::init_ctxt();

	QTemporaryDir dir;
	PV_ASSERT_VALID(dir.isValid(), "a temporary directory", "could not be made");
	const std::string capture = dir.filePath("three_megabytes.pcap").toStdString();
	write_capture(capture, 2000, 1500);

	bool canceled = false;
	const pvpcap::splitted_files_t files = pvpcap::split_pcaps(
	    {capture}, dir.filePath("split").toStdString(), /*preserve_flows=*/false, canceled);
	PV_ASSERT_VALID(not files.empty(), "the capture", "was not split");

	std::cout << "split into " << files.size() << " file(s) with no progress callback"
	          << std::endl;
	return 0;
}
