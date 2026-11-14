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
 */
void check_types(const std::vector<column>& columns, size_t rows)
{
	const QString path_qs =
	    PVCore::mkstemp(PVRush::PVNrawCacheManager::nraw_dir() + "/types_discovery_XXXXXX.csv");
	PV_ASSERT_VALID(not path_qs.isEmpty(), "input file", path_qs.toStdString());
	const std::string path = path_qs.toStdString();
	{
		std::ofstream out(path);
		for (size_t r = 0; r <= rows; r++) {
			for (size_t i = 0; i < columns.size(); i++) {
				const column& c = columns[i];
				out << (i == 0 ? "" : ",")
				    << (r == 0 ? c.name : c.values[(r - 1) % c.values.size()]);
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
		PV_VALID(name, columns[i].name);
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
	check_types({{"amount", amounts, "number_float", ""},
	             {"comment", {std::string(400, 'x')}, "string", ""}},
	            40000);
}

} // namespace

int main()
{
	pvtest::init_ctxt();

	check_dates_read_whole();
	check_types_across_chunks();

	return 0;
}
