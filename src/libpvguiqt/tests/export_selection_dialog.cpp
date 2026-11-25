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

// The formats a selection exports to: CSV, the format of the type of its source if that type has
// an exporter, and those of the other types that export the NRAW of any source, such as Parquet.

#include "common.h"
#include "test-env.h"

#include <pvguiqt/PVExportSelectionDlg.h>

#include <squey/PVRoot.h>
#include <squey/PVScene.h>
#include <squey/PVSelection.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVTaskProgress.h>
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVFileListDescription.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVSourceCreator.h>

#include <QApplication>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRadioButton>
#include <QTemporaryDir>

#include <string>

namespace
{

constexpr const char* PARQUET = ".parquet files (*.parquet)";

constexpr const char* FORMAT = R"(<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE PVParamXml>
<param version="11" first_line="0">
 <splitter type="csv" sep="," quote="&quot;">
  <field><axis name="axis1" type="number_uint32"/></field>
  <field><axis name="axis2" type="number_uint32"/></field>
  <field><axis name="axis3" type="number_uint32"/></field>
 </splitter>
</param>
)";

void write(QString const& path, QByteArray const& content)
{
	QFile file(path);
	PV_ASSERT_VALID(file.open(QIODevice::WriteOnly), "file", path.toStdString());
	file.write(content);
}

Squey::PVView& view_of(Squey::PVSource& source)
{
	return source.emplace_add_child().emplace_add_child().emplace_add_child();
}

/**
 * Imports the rows of the parquet file @p path that @p row_filter keeps, as the main window
 * imports its sample dataset.
 */
Squey::PVSource&
import_parquet(Squey::PVRoot& root, QString const& path, QString const& row_filter = {})
{
	const PVRush::PVInputType_p type =
	    LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("parquet");
	PVRush::PVInputType::list_inputs inputs;
	PVRush::PVFormat format;
	const QString params =
	    QJsonDocument(QJsonObject{{"paths", QJsonArray{path}}}).toJson(QJsonDocument::Compact);
	PVCore::PVTaskProgress progress;
	PV_ASSERT_VALID(type->create_source_description_params(params, inputs, format, progress));
	static_cast<PVRush::PVFileListDescription&>(*inputs.front()).set_row_filter(row_filter);

	Squey::PVSource& source = root.emplace_add_child("parquet").emplace_add_child(
	    inputs, LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("parquet"), format);
	source.wait_extract_end(source.extract(0));
	return source;
}

/** How many choices of the axes to export the pages of @p dialog name @p all_label. */
int choices_of_all(PVGuiQt::PVExportSelectionDlg const& dialog, QString const& all_label)
{
	int count = 0;
	for (auto* button : dialog.findChildren<QRadioButton*>()) {
		count += button->text() == all_label;
	}
	return count;
}

} // namespace

int main(int argc, char** argv)
{
	init_env();

	QTemporaryDir directory;
	PV_ASSERT_VALID(directory.isValid());
	const QString csv = directory.filePath("axes.csv");
	write(csv, "1,2,3\n4,5,6\n");
	write(csv + ".format", FORMAT);

	Squey::PVRoot root;
	Squey::PVView& csv_view = view_of(get_src_from_file(root, csv, csv + ".format"));
	csv_view.set_axes_combination({PVCol(2), PVCol(0)});

	QApplication app(argc, argv); // argv carries "-platform offscreen"

	const QString exported = directory.filePath("exported.parquet");
	{
		PVGuiQt::PVExportSelectionDlg dialog(csv_view);
		PV_VALID(dialog.name_filters().count(PARQUET), qsizetype(1));
		// The pages of the CSV export and of the Parquet one.
		PV_VALID(choices_of_all(dialog, "Use all axes combination"), 2);
		PV_VALID(choices_of_all(dialog, "Use all the columns of the file"), 0);
		// The CSV export stays the default, for a type of source with no exporter of its own.
		PV_ASSERT_VALID(dialog.selected_name_filter().startsWith(".csv"));
		PV_VALID(dialog.exporter().exports_columns(), false);

		dialog.select_name_filter(PARQUET);
		PV_VALID(dialog.exporter().exports_columns(), true);
		Squey::PVSelection selection(csv_view.get_row_count());
		selection.select_all();
		dialog.exporter().export_rows(exported.toStdString(), selection);

		dialog.select_name_filter(dialog.name_filters().last());
		PV_VALID(dialog.exporter().exports_columns(), false);
	}

	// The axes of the combination of the view, as for the CSV export by default.
	Squey::PVSource& parquet_source = import_parquet(root, exported);
	Squey::PVView& parquet_view = view_of(parquet_source);
	PV_VALID(parquet_view.get_axes_combination().get_nraw_names().join(",").toStdString(),
	         std::string("axis3,axis1"));
	PV_VALID(parquet_source.get_rushnraw().at_string(1, PVCol(0)), std::string("6"));

	{
		// The exporter of the type of the source, rather than the one of its NRAW, by default.
		PVGuiQt::PVExportSelectionDlg dialog(parquet_view);
		PV_VALID(dialog.name_filters().count(PARQUET), qsizetype(1));
		PV_VALID(dialog.selected_name_filter().toStdString(), std::string(PARQUET));
		PV_VALID(choices_of_all(dialog, "Use all the columns of the file"), 1);
		PV_VALID(choices_of_all(dialog, "Use all axes combination"), 1);
		PV_VALID(dialog.exporter().exports_columns(), true);
	}

	{
		// Its rows filtered, a parquet source is exported from its NRAW, as one of another type.
		Squey::PVSource& filtered_source = import_parquet(root, exported, "axis3 > 3");
		PV_VALID(size_t(filtered_source.get_rushnraw().row_count()), size_t(1));
		PVGuiQt::PVExportSelectionDlg dialog(view_of(filtered_source));
		PV_VALID(dialog.name_filters().count(PARQUET), qsizetype(1));
		PV_VALID(choices_of_all(dialog, "Use all the columns of the file"), 0);
		PV_VALID(choices_of_all(dialog, "Use all axes combination"), 2);
		dialog.select_name_filter(PARQUET);
		PV_VALID(dialog.exporter().exports_columns(), true);
	}

	return 0;
}
