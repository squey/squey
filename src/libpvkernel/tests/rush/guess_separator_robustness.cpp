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

// The separator and the number of fields guessed for files that are not perfectly
// regular, or that several separators split as regularly.

#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/filter/PVFieldSplitterChunkMatch.h>
#include <pvkernel/core/PVUtils.h>
#include <pvkernel/core/squey_assert.h>

#include "common.h"

#include <cstdio>
#include <fstream>
#include <functional>
#include <string>

namespace
{

/**
 * Guesses the splitter of a file of the header @p header and of 300 lines @p line makes
 * of their number, and checks it splits on @p sep into @p fields fields, or that no
 * splitter is found when @p fields is 0.
 */
void check_guess(const std::string& name,
                 const std::string& header,
                 const std::function<std::string(size_t)>& line,
                 char sep,
                 size_t fields)
{
	const QString path_qs =
	    PVCore::mkstemp(PVRush::PVNrawCacheManager::nraw_dir() + "/guess_separator_XXXXXX.csv");
	PV_ASSERT_VALID(not path_qs.isEmpty(), "input file", path_qs.toStdString());
	const std::string path = path_qs.toStdString();
	{
		std::ofstream out(path);
		if (not header.empty()) {
			out << header << '\n';
		}
		for (size_t i = 0; i < 300; i++) {
			out << line(i) << '\n';
		}
	}

	PVRush::PVInputDescription_p file(new PVRush::PVFileDescription(path_qs));
	PVRush::PVSourceCreator_p sc =
	    LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("text_file");
	PVCol nfields(0);
	PVFilter::PVFieldsSplitter_p splitter = PVFilter::PVFieldSplitterChunkMatch::get_match_on_input(
	    sc->create_source_from_input(file), nfields);

	if (fields == 0) {
		PV_ASSERT_VALID(splitter.get() == nullptr, "file", name);
	} else {
		PV_ASSERT_VALID(splitter.get() != nullptr, "file", name);
		PV_VALID(splitter->get_args().at("sep").toChar().toLatin1(), sep, "file", name);
		PV_VALID((size_t)nfields, fields, "file", name);
	}

	std::remove(path.c_str());
}

std::string n(size_t i)
{
	return std::to_string(i);
}

std::string datetime(size_t i)
{
	return "2024-0" + n(1 + i % 9) + "-1" + n(i % 10) + " 1" + n(i % 10) + ":0" + n(i % 10) + ":00";
}

/**
 * A few lines that do not split like the others leave the separator the others say
 */
void check_irregular_lines()
{
	check_guess(
	    "one unquoted separator too many", "id,name,price",
	    [](size_t i) { return i == 150 ? "149,Smith, John,12.50" : n(i) + ",name,12.50"; }, ',', 3);
	check_guess(
	    "a total at the end", "id,name,qty,price",
	    [](size_t i) { return i == 299 ? std::string("Total,1234.56") : n(i) + ",a,3,1.5"; }, ',',
	    4);
	check_guess(
	    "empty fields left out", "a,b,c,d",
	    [](size_t i) { return i % 7 ? n(i) + ",x," + n(i) + ",y" : n(i) + ",x"; }, ',', 4);
	check_guess(
	    "a comment holding a comma", "# exported by a tool, version 1.2\nid,name,price",
	    [](size_t i) { return n(i) + ",name,12.50"; }, ',', 3);
	check_guess(
	    "a line feed inside quotes", "id,comment,age",
	    [](size_t i) {
		    return i % 10 == 3 ? n(i) + ",\"first line\nsecond line\"," + n(i)
		                       : n(i) + ",\"word\"," + n(i);
	    },
	    ',', 3);
	// a quote that each line opens and nothing closes is a mere character
	check_guess(
	    "quotes that never close", "id,path,n",
	    [](size_t i) { return n(i) + ",\"C:\\dir," + n(i); }, ',', 3);
	// too irregular to be columns
	check_guess(
	    "two record types", "",
	    [](size_t i) { return i % 2 ? n(i) + ",a,b,c" : n(i) + ",a,b,c,d"; }, ',', 0);
}

/**
 * Of two separators splitting every line as regularly, the one whose values are values
 */
void check_even_splits()
{
	check_guess(
	    "dates holding a space", "id,when", [](size_t i) { return n(i) + "," + datetime(i); }, ',',
	    2);
	check_guess(
	    "two words and a date", "id,city,when",
	    [](size_t i) { return n(i) + ",New York," + datetime(i); }, ',', 3);
	check_guess(
	    "decimal commas", "id;price;day",
	    [](size_t i) { return n(i) + ";12,5" + n(i % 10) + ";01/02/2024"; }, ';', 3);
	check_guess(
	    "decimal commas between tabulations", "id\tprice\tqty",
	    [](size_t i) { return n(i) + "\t12,5\t13,7"; }, '\t', 3);
	check_guess(
	    "another separator in every value", "id,type",
	    [](size_t i) { return n(i) + ",text/html; charset=utf-8"; }, ',', 2);
	check_guess(
	    "a space as the separator", "", [](size_t i) { return n(i) + " 12 " + n(i * 2); }, ' ', 3);
	check_guess(
	    "directories quoted", "id,path,n",
	    [](size_t i) { return n(i) + ",\"C:\\dir, " + n(i) + "\\\"," + n(i); }, ',', 3);
}

} // namespace

int main()
{
	pvtest::init_ctxt();

	check_irregular_lines();
	check_even_splits();

	return 0;
}
