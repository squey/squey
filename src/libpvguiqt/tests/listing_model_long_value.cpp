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

// The text a listing gives its view for a long value.
//
// Qt lays out the whole text of a cell to elide it: a value of millions of
// characters took seconds at each repaint, and froze the listing. The listing gives
// at most DISPLAYED_SIZE_MAX bytes of a value, cut between two characters and
// followed by an ellipsis, in the cell as in its tooltip.

#include <pvkernel/core/squey_assert.h>

#include <squey/PVRoot.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvguiqt/PVListingModel.h>

#include <QApplication>
#include <QTemporaryDir>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "test-env.h"

namespace
{

constexpr const char* FORMAT = R"(<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE PVParamXml>
<param version="11" first_line="0">
 <splitter type="csv" sep="," quote="&quot;">
  <field>
   <axis name="value" type="string">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
 </splitter>
</param>
)";

constexpr size_t MAX = PVGuiQt::PVAbstractTableModel::DISPLAYED_SIZE_MAX;
constexpr QChar ELLIPSIS(0x2026);

// Four bytes in UTF-8: after 0 to 3 ASCII characters, the cut falls on each of its
// bytes in turn.
constexpr const char* EMOJI = "\xF0\x9F\x98\x80";

struct Value {
	std::string imported;
	QString displayed;
};

std::vector<Value> expected_values()
{
	std::vector<Value> values{
	    {"short value", "short value"},
	    {std::string(MAX, 'a'), QString(MAX, u'a')},
	    {std::string(MAX + 1, 'b'), QString(MAX, u'b') + ELLIPSIS},
	    {std::string(size_t(1) << 23, 'c'), QString(MAX, u'c') + ELLIPSIS},
	};
	for (size_t lead = 0; lead < 4; ++lead) {
		std::string imported(lead, 'd');
		for (size_t i = 0; i <= MAX / 4; ++i) {
			imported += EMOJI;
		}
		values.push_back(
		    {imported,
		     QString(lead, u'd') + QString::fromUtf8(EMOJI).repeated((MAX - lead) / 4) + ELLIPSIS});
	}
	return values;
}

} // namespace

int main(int argc, char** argv)
{
	init_env();

	const std::vector<Value> expected = expected_values();

	QTemporaryDir dir;
	PV_ASSERT_VALID(dir.isValid(), "a temporary directory", "could not be made");
	const QString csv = dir.filePath("values.csv");
	const QString format = dir.filePath("values.csv.format");
	{
		std::ofstream out(csv.toStdString(), std::ios::binary);
		// A byte order mark settles the charset, otherwise guessed from the first chunk:
		// filled with the 8 MiB of ASCII, it passed for ISO-8859-9.
		out << "\xEF\xBB\xBF";
		for (const Value& value : expected) {
			out << value.imported << "\n";
		}
		std::ofstream(format.toStdString()) << FORMAT;
	}

	auto root = std::make_unique<Squey::PVRoot>();
	Squey::PVSource& src = get_src_from_file(*root, csv, format);
	src.emplace_add_child()   // Mapped
	    .emplace_add_child()  // Scaled
	    .emplace_add_child(); // View
	Squey::PVView& view = *src.current_view();

	QApplication app(argc, argv);

	PVGuiQt::PVListingModel model(view);
	PV_ASSERT_VALID(model.rowCount() == int(expected.size()), "rows listed", model.rowCount());

	for (int row = 0; row < int(expected.size()); ++row) {
		const QModelIndex index = model.index(row, 0);
		const QString displayed = model.data(index, Qt::DisplayRole).toString();
		const QString tooltip = model.data(index, Qt::ToolTipRole).toString();
		std::cout << "row " << row << ": " << expected[row].imported.size() << " bytes imported, "
		          << displayed.size() << " UTF-16 code units displayed, " << tooltip.size()
		          << " in the tooltip" << std::endl;

		PV_ASSERT_VALID(displayed == expected[row].displayed, "row", row, "expected code units",
		                expected[row].displayed.size());
		PV_ASSERT_VALID(QString(tooltip).remove("<br>") == expected[row].displayed, "row", row,
		                "expected code units", expected[row].displayed.size());
	}

	return 0;
}
