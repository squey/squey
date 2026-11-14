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

// The lines a text source cuts a file into, across the chunks it reads it in.

#include <pvkernel/rush/PVInputFile.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/rush/PVUnicodeSource.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVTextChunk.h>
#include <pvkernel/core/PVUtils.h>
#include <pvkernel/core/squey_assert.h>

#include "common.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

static constexpr size_t chunk_size = 4096;

template <template <class T> class Allocator>
static std::vector<std::string> lines_read(const std::string& path,
                                           std::optional<PVRush::PVCSVQuoting> quoting = {})
{
	PVRush::PVUnicodeSource<Allocator> source(std::make_shared<PVRush::PVInputFile>(path.c_str()),
	                                          chunk_size);
	source.set_quoting(quoting);
	std::vector<std::string> lines;
	while (PVCore::PVChunk* chunk = source()) {
		for (PVCore::PVElement* line : static_cast<PVCore::PVTextChunk*>(chunk)->c_elements()) {
			lines.emplace_back(line->begin(), line->end());
		}
		chunk->free();
	}
	return lines;
}

static void check_lines(const std::vector<std::string>& read,
                        const std::vector<std::string>& expected)
{
	PV_VALID(read.size(), expected.size());
	for (size_t i = 0; i < expected.size(); i++) {
		PV_VALID(read[i], expected[i], "line", i);
	}
}

static std::string written(const std::vector<std::string>& lines, const char* last_end = "\n")
{
	const QString path =
	    PVCore::mkstemp(PVRush::PVNrawCacheManager::nraw_dir() + "/unicode_source_XXXXXX.csv");
	PV_ASSERT_VALID(not path.isEmpty(), "input file", path.toStdString());
	std::ofstream out(path.toStdString(), std::ios::binary);
	for (size_t i = 0; i < lines.size(); i++) {
		out << lines[i] << (i + 1 == lines.size() ? last_end : "\n");
	}
	return path.toStdString();
}

/**
 * The charset is detected on the first chunk alone. One of pure ASCII says nothing
 * against the UTF-8 that follows it, which has to read back as it was written.
 */
static void check_utf8_after_an_ascii_chunk()
{
	std::vector<std::string> lines;
	for (size_t i = 0; i < 4 * chunk_size / 32; i++) {
		lines.push_back(std::to_string(i) + ",plain,ascii only text");
	}
	for (const char* name : {"café", "Zoë", "Ελλάδα", "東京", "naïve 😀"}) {
		lines.push_back(std::to_string(lines.size()) + "," + name + ",after the first chunk");
	}
	const std::string path = written(lines);

	check_lines(lines_read<PVCore::PVMMapAllocator>(path), lines);

	std::remove(path.c_str());
}

/**
 * Memory an allocator hands out again holds what was there before: line feeds, say,
 * from the chunks of another file.
 */
template <class T>
struct recycled_allocator : std::allocator<T> {
	template <class U>
	struct rebind {
		using other = recycled_allocator<U>;
	};

	recycled_allocator() = default;
	template <class U>
	explicit recycled_allocator(const recycled_allocator<U>&)
	{
	}

	T* allocate(size_t n)
	{
		T* p = std::allocator<T>::allocate(n);
		std::memset(static_cast<void*>(p), '\n', n * sizeof(T));
		return p;
	}
};

/**
 * The lines are cut at the last line feed the data read holds, wherever the chunks end
 */
static void check_line_ends_in_recycled_memory()
{
	std::vector<std::string> lines;
	for (size_t i = 0; i < 8 * chunk_size / 32; i++) {
		lines.push_back(std::to_string(i) + "," + std::string(i % 61, 'x'));
	}
	const std::string path = written(lines);

	check_lines(lines_read<recycled_allocator>(path), lines);

	std::remove(path.c_str());
}

/**
 * The lines of another charset read back in UTF-8, the last one too, which no line feed
 * ends, when the conversion outgrows the chunks
 */
static void check_converted_lines()
{
	std::vector<std::string> latin1;
	std::vector<std::string> utf8;
	for (size_t i = 0; i < 8 * chunk_size / 32; i++) {
		latin1.push_back(std::to_string(i) + ",d\xe9j\xe0 vu \xe0 l'\xe9t\xe9," +
		                 std::string(i % 13, '\xe9'));
		std::string accents;
		for (size_t a = 0; a < i % 13; a++) {
			accents += "é";
		}
		utf8.push_back(std::to_string(i) + ",déjà vu à l'été," + accents);
	}
	const std::string path = written(latin1, "");

	check_lines(lines_read<PVCore::PVMMapAllocator>(path), utf8);
	check_lines(lines_read<recycled_allocator>(path), utf8);

	std::remove(path.c_str());
}

/**
 * The quoted values of a CSV hold line feeds, wherever the chunks end, also once converted
 * from another charset
 */
static void check_quoted_line_feeds()
{
	const PVRush::PVCSVQuoting csv{',', '"'};

	std::vector<std::string> records;
	for (size_t i = 0; i < 8 * chunk_size / 32; i++) {
		const std::string text =
		    std::string(i % 37, 'x') + "\n" + std::string(i % 23, 'y') + (i % 3 ? "" : "\"\"");
		records.push_back(std::to_string(i) + ",\"" + text + "\",plain," +
		                  (i % 5 ? "z" : "\"\"\"\r\n\""));
	}
	records.emplace_back("last,\"one\nwithout a line feed after\"");
	const std::string path = written(records, "");

	check_lines(lines_read<PVCore::PVMMapAllocator>(path, csv), records);

	// a file the first read holds whole, whose last quote ends it
	for (const std::vector<std::string>& small :
	     {std::vector<std::string>{"\"a\",\"b\""}, {"1,2", "3,\"x\ny\""}}) {
		const std::string small_path = written(small, "");
		check_lines(lines_read<PVCore::PVMMapAllocator>(small_path, csv), small);
		std::remove(small_path.c_str());
	}

	// the lines of the same file, without quoting
	std::vector<std::string> lines;
	for (const std::string& record : records) {
		for (size_t begin = 0, end; begin != std::string::npos; begin = end) {
			end = record.find('\n', begin);
			lines.push_back(record.substr(begin, end - begin));
			end = end == std::string::npos ? end : end + 1;
		}
	}
	for (std::string& line : lines) {
		if (not line.empty() and line.back() == '\r') {
			line.pop_back();
		}
	}
	check_lines(lines_read<PVCore::PVMMapAllocator>(path), lines);

	std::remove(path.c_str());

	// in ISO-8859-1, which the source converts to UTF-8
	std::vector<std::string> latin1;
	std::vector<std::string> utf8;
	for (size_t i = 0; i < 4 * chunk_size / 32; i++) {
		latin1.push_back(std::to_string(i) +
		                 ",\"la r\xe9union a eu lieu\npr\xe8s du caf\xe9\",d\xe9j\xe0 vu");
		utf8.push_back(std::to_string(i) + ",\"la réunion a eu lieu\nprès du café\",déjà vu");
	}
	latin1.emplace_back("fin,\"derni\xe8re ligne\nsans fin de ligne\"");
	utf8.emplace_back("fin,\"dernière ligne\nsans fin de ligne\"");
	const std::string latin1_path = written(latin1, "");

	check_lines(lines_read<PVCore::PVMMapAllocator>(latin1_path, csv), utf8);

	std::remove(latin1_path.c_str());
}

int main()
{
	pvtest::init_ctxt();

	check_utf8_after_an_ascii_chunk();
	check_line_ends_in_recycled_memory();
	check_converted_lines();
	check_quoted_line_feeds();

	return 0;
}
