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

// The types the autodetection gives the columns of a CSV, from their values.

#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVFormat_types.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/rush/PVTests.h>
#include <pvkernel/rush/PVTypesDiscoveryOutput.h>
#include <pvkernel/core/PVUtils.h>
#include <pvkernel/core/squey_assert.h>

#include "common.h"

#include <QDomDocument>

#include <cinttypes>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace
{

struct column {
	std::string name;
	std::vector<std::string> values;
	std::string type;
	std::string type_format;
};

// a value that ends its line there, leaving the next fields out
const std::string line_ends_here = "\x01";

QDomDocument csv_format(size_t columns)
{
	QDomDocument doc;
	doc.setContent(
	    QString("<!DOCTYPE PVParamXml><param version=\"" PVFORMAT_CURRENT_VERSION "\"/>"));
	QDomElement splitter = doc.createElement("splitter");
	splitter.setAttribute("type", "csv");
	splitter.setAttribute("sep", ",");
	splitter.setAttribute("quote", "\"");
	doc.documentElement().appendChild(splitter);
	for (size_t i = 0; i < columns; i++) {
		QDomElement field = doc.createElement("field");
		QDomElement axis = doc.createElement("axis");
		axis.setAttribute("type", "string");
		axis.setAttribute("name", QString("Axis %1").arg(i + 1));
		field.appendChild(axis);
		splitter.appendChild(field);
	}
	return doc;
}

/**
 * Runs the autodetection on a CSV of @p rows rows, the column @p c of row @p r holding
 * the value @p r of the column, round and round, and checks the type of each column.
 *
 * @param header whether the names of the columns come first, which the autodetection has
 *               to tell, or the file only holds values
 */
void check_types(const std::vector<column>& columns, size_t rows, bool header = true)
{
	const QString path_qs =
	    PVCore::mkstemp(PVRush::PVNrawCacheManager::nraw_dir() + "/types_discovery_XXXXXX.csv");
	PV_ASSERT_VALID(not path_qs.isEmpty(), "input file", path_qs.toStdString());
	const std::string path = path_qs.toStdString();
	{
		std::ofstream out(path);
		for (size_t r = header ? 0 : 1; r <= rows; r++) {
			for (size_t i = 0; i < columns.size(); i++) {
				const column& c = columns[i];
				const std::string& value = r == 0 ? c.name : c.values[(r - 1) % c.values.size()];
				if (value == line_ends_here) {
					break;
				}
				out << (i == 0 ? "" : ",") << value;
			}
			out << '\n';
		}
	}

	const QDomDocument doc = csv_format(columns.size());
	PVRush::PVFormat format(doc.documentElement());
	PVRush::PVInputDescription_p input(new PVRush::PVFileDescription(path_qs));
	PVRush::PVSourceCreator_p sc;
	PVRush::PVTests::get_file_sc(input, format, sc);
	PVRush::PVTypesDiscoveryOutput types;
	{
		QList<std::shared_ptr<PVRush::PVInputDescription>> inputs;
		inputs << input;
		PVRush::PVExtractor extractor(format, types, sc, inputs);
		extractor.process_from_agg_idxes(0, rows)->wait_end();
	}

	for (size_t i = 0; i < columns.size(); i++) {
		const auto [type, type_format, name] = types.type_desc(PVCol(i));
		PV_VALID(name, header ? columns[i].name : std::string(), "column", columns[i].name);
		PV_VALID(type, columns[i].type, "column", columns[i].name);
		PV_VALID(type_format, columns[i].type_format, "column", columns[i].name);
	}

	std::remove(path.c_str());
}

/**
 * A format that reads only the beginning of the values is not their type: a year of
 * four digits does not read the dates written day first, nor the two digit years, and
 * a time on twelve hours does not read the seconds.
 */
void check_dates_read_whole()
{
	check_types(
	    {
	        {"day_first", {"22/12/2024", "13/11/2024", "24/10/2024"}, "time", "d/M/yyyy"},
	        {"day_first_time",
	         {"13/11/2024 04:17:22", "24/02/2024 12:02:08"},
	         "time",
	         "d/M/yyyy H:m:s"},
	        {"morning", {"2024-05-08 10:34:08", "2024-05-12 02:38:00"}, "time", "yyyy-M-d H:m:s"},
	        {"short_year", {"19/02/14 15:55:47", "03/04/24 10:00:00"}, "time", "d/M/yy H:m:s"},
	    },
	    100);
}

/**
 * Each chunk of the sample is read on its own. A column of integers and decimals has to
 * keep a number type whatever value starts each chunk, and however the chunks overlap.
 */
void check_types_across_chunks()
{
	const std::vector<std::string> amounts{"12", "7", "30", "4", "15", "9", "22", "3", "18", "5.5"};
	// about 16 MB, so that the sample spans two chunks of the default ten
	check_types({{"amount", amounts, "number_double", "%.15g"},
	             {"comment", {std::string(400, 'x')}, "string", ""}},
	            40000);
}

/**
 * The decimals are read in a type that keeps them as written
 */
void check_decimals()
{
	check_types({{"price", {"25.01", "590.49", "716.02"}, "number_double", "%.15g"},
	             {"big", {"4326096.668", "2532120.958"}, "number_double", "%.15g"},
	             {"scientific", {"-9.510e+08", "1e-5", "-inf"}, "number_double", "%.15g"}},
	            100);
}

/**
 * A number type only takes the values it writes back as they are written
 */
void check_number_notations()
{
	check_types({{"zip", {"01234", "06000", "75001", "13008"}, "string", ""},
	             {"phone", {"0634825892", "0738328553"}, "string", ""},
	             {"word", {"cafe", "bad", "face", "d"}, "string", ""},
	             {"hexadecimal", {"0x1F", "0xdeadbeef"}, "number_uint32", "0x%" PRIx32},
	             {"zero", {"0", "10", "-3"}, "number_int32", ""},
	             {"fraction", {"0.5", "-0.25"}, "number_double", "%.15g"}},
	            100);

	// a header holding a word an hexadecimal number reads is still a header
	check_types({{"d", {"2024-02-23 10:00:00"}, "time", "yyyy-M-d H:m:s"},
	             {"name", {"alpha", "bravo"}, "string", ""}},
	            100);
}

/**
 * The usual ways of writing that a field has no value rule no type out, and a column
 * holding nothing else is text
 */
void check_missing_values()
{
	check_types(
	    {{"qty", {"12", "NULL", "7", "N/A", "-", "3", "null", " NA "}, "number_uint32", ""},
	     {"price", {"2.9", "NaN", "1.5", "#N/A"}, "number_double", "%.15g"},
	     {"when", {"2024-01-01 10:00:00", "-", "2024-01-02 11:00:00"}, "time", "yyyy-M-d H:m:s"},
	     {"dashes", {"-"}, "string", ""},
	     {"nothing", {""}, "string", ""}},
	    100);
}

/**
 * The dates of ISO 8601, of the logs of web servers and of mails, and the dates written
 * month first, which only read so when they do not read day first
 */
void check_date_formats()
{
	check_types(
	    {
	        {"iso_date", {"2024-02-23", "2024-12-01"}, "time", "yyyy-M-d"},
	        {"iso", {"2024-01-28T10:25:00", "2024-01-03T10:53:00"}, "time", "yyyy-M-d'T'H:m:s"},
	        {"iso_offset",
	         {"2024-01-15T10:00:00+01:00", "2024-01-15T10:00:00Z"},
	         "time",
	         "yyyy-M-d'T'H:m:sX"},
	        {"iso_fraction",
	         {"2024-01-16T10:00:00.123Z", "2024-01-16T10:00:00.123456+02:00"},
	         "time",
	         "yyyy-M-d H:m:s.SX"},
	        {"postgresql", {"2024-01-01 10:00:00.123456+01"}, "time", "yyyy-M-d H:m:s.SX"},
	        {"month_first", {"09/28/2024", "02/13/2024"}, "time", "M/d/yyyy"},
	        {"either_way", {"03/04/2024", "11/12/2024"}, "time", "d/M/yyyy"},
	        {"web_server", {"05/Jan/2024:10:53:00 +0000"}, "time", "d/MMM/yyyy:H:m:s Z"},
	        {"mail", {"Mon, 01 Jan 2024 10:00:00 +0000"}, "time", "eee, d MMM yyyy H:m:s Z"},
	    },
	    100);
}

/**
 * The first row names the columns when it holds no number or date, or when a column of
 * numbers or dates does not read what it holds there, which then rules no type out
 */
void check_header()
{
	check_types({{"id", {"1", "2", "3"}, "number_uint32", ""},
	             {"2023", {"1.5", "2.5"}, "number_double", "%.15g"},
	             {"2024", {"3.5", "4.5"}, "number_double", "%.15g"}},
	            100);
	check_types({{"", {"1", "2", "3"}, "number_uint32", ""},
	             {"name", {"alpha", "bravo"}, "string", ""},
	             {"when", {"2024-01-01 10:00:00"}, "time", "yyyy-M-d H:m:s"}},
	            100);

	// values only, the first row as the others
	check_types({{"id", {"1", "2", "3"}, "number_uint32", ""},
	             {"price", {"1.5", "2.5"}, "number_double", "%.15g"},
	             {"name", {"alpha", "bravo"}, "string", ""}},
	            100, false);
	check_types({{"qty", {"NULL", "2", "3"}, "number_uint32", ""},
	             {"price", {"1.5", "2.5"}, "number_double", "%.15g"}},
	            100, false);
}

/**
 * The fields a line lacks are empty, and rule no type out
 */
void check_short_lines()
{
	check_types({{"id", {"1", "2", "3"}, "number_uint32", ""},
	             {"qty", {"4", "5", line_ends_here}, "number_uint32", ""},
	             {"price", {"1.5", "2.5", "9.5"}, "number_double", "%.15g"}},
	            100);
}

} // namespace

int main()
{
	pvtest::init_ctxt();

	check_dates_read_whole();
	check_types_across_chunks();
	check_decimals();
	check_number_notations();
	check_missing_values();
	check_date_formats();
	check_header();
	check_short_lines();

	return 0;
}
