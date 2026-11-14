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
#include <string>
#include <vector>

static constexpr size_t chunk_size = 4096;

template <template <class T> class Allocator>
static std::vector<std::string> lines_read(const std::string& path)
{
	PVRush::PVUnicodeSource<Allocator> source(std::make_shared<PVRush::PVInputFile>(path.c_str()),
	                                          chunk_size);
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

int main()
{
	pvtest::init_ctxt();

	check_utf8_after_an_ascii_chunk();
	check_line_ends_in_recycled_memory();
	check_converted_lines();

	return 0;
}
