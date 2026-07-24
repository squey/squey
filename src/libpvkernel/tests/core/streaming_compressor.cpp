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

#include <pvkernel/core/PVStreamingCompressor.h>
#include <pvkernel/core/squey_assert.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static fs::path TEST_DIR;

static std::string make_content()
{
	// several megabytes to exercise chunked pipe I/O, with enough entropy to
	// require multiple compressed chunks
	std::string content;
	std::mt19937 gen(42);
	for (size_t i = 0; i < 200'000; i++) {
		content += "line " + std::to_string(i) + " value " + std::to_string(gen()) + "\n";
	}
	return content;
}

static void compress_file(const std::string& path, const std::string& content, size_t chunk_size)
{
	PVCore::PVStreamingCompressor compressor(path);
	for (size_t offset = 0; offset < content.size(); offset += chunk_size) {
		compressor.write(content.substr(offset, chunk_size));
	}
	compressor.wait_finished();
}

static std::string decompress_file(const std::string& path)
{
	PVCore::PVStreamingDecompressor decompressor(path);
	std::string result;
	char buffer[64 * 1024];
	while (true) {
		auto [uncompressed, compressed] = decompressor.read(buffer, sizeof(buffer));
		if (uncompressed == 0) {
			break;
		}
		result.append(buffer, uncompressed);
	}
	return result;
}

static bool tool_available(const std::string& extension)
{
	const auto args = PVCore::PVStreamingDecompressor::executable(
	    extension, PVCore::PVStreamingDecompressor::EExecType::DECOMPRESSOR);
	if (args.empty()) {
		return false;
	}
#ifdef _WIN32
	const std::string cmd = "where " + args[0] + " > nul 2>&1";
#else
	const std::string cmd = "command -v " + args[0] + " > /dev/null 2>&1";
#endif
	return std::system(cmd.c_str()) == 0;
}

static void test_round_trip(const std::string& filename, const std::string& content)
{
	const std::string path = (TEST_DIR / filename).string();
	// Announced separately: a compressor dying mid-write takes the whole process
	// with it, so the last line printed is the only clue left as to which half
	// of the round trip broke.
	std::cout << "  compressing " << filename << std::endl;
	compress_file(path, content, 1024 * 1024);
	std::cout << "  decompressing " << filename << std::endl;
	PV_VALID(decompress_file(path) == content, true);
	fs::remove(path);
}

static void test_passthrough(const std::string& content)
{
	std::cout << "passthrough round-trip + reset" << std::endl;
	const std::string path = (TEST_DIR / "data.txt").string();
	compress_file(path, content, 1024 * 1024);

	// the decompressor is scoped: Windows refuses to remove a file that is still open
	{
		PVCore::PVStreamingDecompressor decompressor(path);
		char buffer[64 * 1024];
		auto [uncompressed, compressed] = decompressor.read(buffer, sizeof(buffer));
		PV_VALID(uncompressed > 0UL, true);
		PV_VALID(uncompressed, compressed);

		// reset() must rewind to the beginning of the file
		decompressor.reset();
		std::string result;
		while (true) {
			auto [count, _] = decompressor.read(buffer, sizeof(buffer));
			if (count == 0) {
				break;
			}
			result.append(buffer, count);
		}
		PV_VALID(result == content, true);
	}
	fs::remove(path);
}

static void test_no_extension_is_passthrough()
{
	std::cout << "file named after an extension is not decompressed" << std::endl;
	// a file whose *name* is "gz" (no dot) must not be treated as gzip-compressed
	const std::string path = (TEST_DIR / "gz").string();
	const std::string content = "plain text content\n";
	{
		PVCore::PVStreamingCompressor compressor(path);
		compressor.write(content);
		compressor.wait_finished();
	}
	std::string on_disk;
	{
		std::ifstream ifs(path, std::ios::binary);
		on_disk.assign((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	}
	PV_VALID(on_disk == content, true); // written verbatim, not compressed
	PV_VALID(decompress_file(path) == content, true);
	fs::remove(path);
}

static void test_existing_file_is_truncated()
{
	std::cout << "exporting over a longer existing file truncates it" << std::endl;
	const std::string path = (TEST_DIR / "truncated.txt").string();
	{
		std::ofstream ofs(path, std::ios::binary);
		ofs << std::string(4096, 'o'); // stale content from a previous export
	}
	const std::string content = "new content\n";
	{
		PVCore::PVStreamingCompressor compressor(path);
		compressor.write(content);
		compressor.wait_finished();
	}
	PV_VALID(fs::file_size(path) == content.size(), true);
	fs::remove(path);
}

static void test_uppercase_extension(const std::string& content)
{
	std::cout << "uppercase extension is recognized" << std::endl;
	test_round_trip("data.GZ", content);
}

static void test_filename_with_spaces(const std::string& content)
{
	std::cout << "zip file name containing spaces" << std::endl;
	test_round_trip("file name with spaces.zip", content);
}

static void test_compressor_invalid_path()
{
	std::cout << "compressor on invalid path throws" << std::endl;
	bool thrown = false;
	try {
		PVCore::PVStreamingCompressor compressor((TEST_DIR / "no_such_dir" / "f.gz").string());
	} catch (const PVCore::PVStreamingCompressorError&) {
		thrown = true;
	}
	PV_VALID(thrown, true);
}

static void test_decompressor_missing_file()
{
	std::cout << "decompressor on missing file throws" << std::endl;
	PVCore::PVStreamingDecompressor decompressor((TEST_DIR / "missing.gz").string());
	bool thrown = false;
	char buffer[16];
	try {
		decompressor.read(buffer, sizeof(buffer));
	} catch (const PVCore::PVStreamingDecompressorError&) {
		thrown = true;
	}
	PV_VALID(thrown, true);
}

static void test_corrupted_input()
{
	std::cout << "corrupted compressed file reports an error (no hang, no crash)" << std::endl;
	const std::string path = (TEST_DIR / "corrupted.gz").string();
	{
		// large enough for the feeder thread to still be writing when the
		// decompression process dies, which is how the failure is detected on
		// platforms where the child exit code is not checked
		std::ofstream ofs(path, std::ios::binary);
		const std::string garbage(1024 * 1024, 'x');
		for (size_t i = 0; i < 8; i++) {
			ofs << garbage;
		}
	}
	bool thrown = false;
	{
		PVCore::PVStreamingDecompressor decompressor(path);
		char buffer[64 * 1024];
		try {
			while (true) {
				auto [count, _] = decompressor.read(buffer, sizeof(buffer));
				if (count == 0) {
					break;
				}
			}
		} catch (const PVCore::PVStreamingDecompressorError& e) {
			std::cout << "  got expected error: " << e.what() << std::endl;
			thrown = true;
		}
	}
	PV_VALID(thrown, true);
	fs::remove(path);
}

static void test_decompressor_cancel_and_reset(const std::string& content)
{
	std::cout << "decompressor cancel in the middle, then reset and full read" << std::endl;
	const std::string path = (TEST_DIR / "cancel.gz").string();
	compress_file(path, content, 1024 * 1024);

	{
		PVCore::PVStreamingDecompressor decompressor(path);
		char buffer[4096];
		auto [count, _] = decompressor.read(buffer, sizeof(buffer));
		PV_VALID(count > 0UL, true);
		decompressor.reset(); // cancels the in-flight decompression

		std::string result;
		while (true) {
			auto [c, __] = decompressor.read(buffer, sizeof(buffer));
			if (c == 0) {
				break;
			}
			result.append(buffer, c);
		}
		PV_VALID(result == content, true);
	}
	fs::remove(path);
}

static void test_compressor_cancel()
{
	std::cout << "compressor cancel does not throw nor hang" << std::endl;
	const std::string path = (TEST_DIR / "canceled.gz").string();
	{
		PVCore::PVStreamingCompressor compressor(path);
		compressor.write("some content\n");
		compressor.cancel();
	}
	fs::remove(path);
}

static void test_compressor_destructor_during_unwind()
{
	std::cout << "compressor destroyed during exception unwinding cleans up silently" << std::endl;
	const std::string path = (TEST_DIR / "unwind.gz").string();
	bool caught = false;
	try {
		PVCore::PVStreamingCompressor compressor(path);
		compressor.write("some content\n");
		throw std::runtime_error("simulated export failure");
	} catch (const std::runtime_error&) {
		caught = true;
	}
	PV_VALID(caught, true);
	fs::remove(path);
}

static void test_compression_failure_reports_error()
{
	if (not fs::exists("/dev/full")) {
		std::cout << "skipping compression failure test (no /dev/full)" << std::endl;
		return;
	}
	std::cout << "compression to a full device reports an error (no hang)" << std::endl;
	const std::string path = (TEST_DIR / "full.gz").string();
	fs::create_symlink("/dev/full", path);
	bool thrown = false;
	try {
		PVCore::PVStreamingCompressor compressor(path);
		const std::string chunk(1024 * 1024, 'x');
		for (size_t i = 0; i < 64; i++) {
			compressor.write(chunk);
		}
		compressor.wait_finished();
	} catch (const PVCore::PVStreamingCompressorError& e) {
		std::cout << "  got expected error: " << e.what() << std::endl;
		thrown = true;
	}
	PV_VALID(thrown, true);
	fs::remove(path);
}

static void test_supported_extensions()
{
	std::cout << "supported extensions" << std::endl;
	const auto extensions = PVCore::PVStreamingCompressor::supported_extensions();
	for (const char* ext : {"zip", "gz", "bz2", "xz", "zst"}) {
		PV_VALID(std::find(extensions.begin(), extensions.end(), ext) != extensions.end(), true);
	}
}

int main()
{
	// Flush every step as it is announced: stdout is a pipe under ctest, so the
	// default block buffering would drop the tail if this ever died on a signal,
	// leaving the log stopping well before the step that actually failed.
	std::cout << std::unitbuf;

	TEST_DIR = fs::temp_directory_path() /
	           ("squey_streaming_compressor_" + std::to_string(std::random_device{}()));
	fs::create_directories(TEST_DIR);

	const std::string content = make_content();

	test_supported_extensions();
	test_passthrough(content);
	test_no_extension_is_passthrough();
	test_existing_file_is_truncated();
	test_compressor_invalid_path();
	test_decompressor_missing_file();
	test_compressor_cancel();
	test_compressor_destructor_during_unwind();

	for (const std::string& ext : PVCore::PVStreamingCompressor::supported_extensions()) {
		if (not tool_available(ext)) {
			std::cout << "skipping ." << ext << " round-trip (tool not available)" << std::endl;
			continue;
		}
		std::cout << "round-trip ." << ext << std::endl;
		test_round_trip("data." + ext, content);
	}

	if (tool_available("gz")) {
		test_uppercase_extension(content);
		test_corrupted_input();
		test_decompressor_cancel_and_reset(content);
		test_compression_failure_reports_error();
	}
	if (tool_available("zip")) {
		test_filename_with_spaces(content);
	}

	fs::remove_all(TEST_DIR);

	return 0;
}
