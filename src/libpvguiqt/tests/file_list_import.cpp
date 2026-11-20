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

// The import of files that an input type reads one after the other, as the import page asks for
// it: the files, read out of the GUI thread for their columns, which the user checks knowing how
// often they have a value, a condition on the rows, checked as it is written, and whether the
// files make one source or one each. Parameters, such as those of D-Bus, choose the same.

#include "common.h"
#include "test-env.h"

#include <squey/PVRoot.h>
#include <squey/PVScene.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvguiqt/PVExportSelectionDlg.h>
#include <pvguiqt/PVImportPage.h>

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVTaskProgress.h>
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVFileListDescription.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/widgets/PVImportFilterWidget.h>
#include <pvkernel/widgets/PVImportWidget.h>

#include <QApplication>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QPlainTextEdit>
#include <QRadioButton>
#include <QTemporaryDir>
#include <QTest>
#include <QTreeWidget>

#include <string>
#include <vector>

namespace
{

constexpr const char* CSV_FORMAT = R"(<?xml version='1.0' encoding='UTF-8'?>
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

/** The content of the import page for files of @p type, once it read @p paths. */
PVWidgets::PVImportWidget&
content_of(PVGuiQt::PVImportPage& page, const PVRush::PVInputType_p& type, QStringList const& paths)
{
	PVWidgets::PVImportWidget* content = type->create_import_widget(nullptr);
	page.set_content(content, "Import", {}, {}, -1);
	content->set_paths(paths);
	PV_ASSERT_VALID(QTest::qWaitFor([content]() { return content->is_ready(); }, 30000));
	return *content;
}

QTreeWidgetItem* item_of(QWidget& content, QString const& column)
{
	const QList<QTreeWidgetItem*> items =
	    content.findChild<QTreeWidget*>("import_columns")->findItems(column, Qt::MatchExactly);
	PV_VALID(items.size(), qsizetype(1), "column", column.toStdString());
	return items.front();
}

std::string fill_rate_of(QWidget& content, QString const& column)
{
	return item_of(content, column)->text(2).toStdString();
}

/** Writes @p condition, then waits for it to be checked, as when the user stops writing it. */
void set_row_filter(QWidget& content, QString const& condition)
{
	content.findChild<QPlainTextEdit*>()->setPlainText(condition);
	auto* filter = content.findChild<PVWidgets::PVImportFilterWidget*>();
	PV_ASSERT_VALID(not filter->is_valid());
	QLabel* error = content.findChild<QLabel*>("row_filter_error");
	PV_ASSERT_VALID(QTest::qWaitFor(
	    [&]() { return filter->is_valid() or not error->text().isEmpty(); }, 10000));
}

std::string row_filter_error_of(QWidget& content)
{
	return content.findChild<QLabel*>("row_filter_error")->text().toStdString();
}

// The type as a char const*: GCC takes a QString made for the call, a temporary, for what the
// reference returned refers to (-Wdangling-reference), which fails the build of Windows.
Squey::PVSource& source_of(Squey::PVRoot& root,
                           char const* type,
                           PVRush::PVInputType::list_inputs const& inputs,
                           PVRush::PVFormat const& format)
{
	Squey::PVSource& source =
	    root.emplace_add_child(std::string(type))
	        .emplace_add_child(
	            inputs, LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name(type), format);
	source.wait_extract_end(source.extract(0));
	return source;
}

std::string axes_of(PVRush::PVFormat const& format)
{
	std::string axes;
	for (PVRush::PVAxisFormat const& axis : format.get_axes()) {
		axes += (axes.empty() ? "" : ",") + axis.get_name().toStdString() + ":" +
		        axis.get_type().toStdString();
	}
	return axes;
}

const PVRush::PVFileListDescription& description_of(PVRush::PVInputType::list_inputs const& inputs)
{
	return static_cast<const PVRush::PVFileListDescription&>(*inputs.front());
}

} // namespace

int main(int argc, char** argv)
{
	init_env();
	QApplication app(argc, argv); // argv carries "-platform offscreen"

	QTemporaryDir directory;
	PV_ASSERT_VALID(directory.isValid());
	const QString first = directory.filePath("first.jsonl");
	const QString second = directory.filePath("second.ndjson");
	write(first, R"({"x": 1, "y": {"z": "a"}, "t": "2024-01-02T03:04:05Z"})"
	             "\n");
	write(second, R"({"x": 2})"
	              "\n"
	              R"({"x": 3, "y": {"z": "b"}})"
	              "\n");
	const PVRush::PVInputType_p json =
	    LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("json");
	Squey::PVRoot root;
	PVGuiQt::PVImportPage page;
	page.show();

	{
		PVWidgets::PVImportWidget& content = content_of(page, json, {first, second});
		PV_VALID(fill_rate_of(content, "x"), std::string("100 %"));
		PV_VALID(fill_rate_of(content, "y.z"), std::string("67 %"));
		PV_VALID(fill_rate_of(content, "t"), std::string("33 %"));
		item_of(content, "y.z")->setCheckState(0, Qt::Unchecked);

		set_row_filter(content, "nope > 1");
		PV_ASSERT_VALID(not content.is_ready());
		PV_ASSERT_VALID(row_filter_error_of(content).find("nope") != std::string::npos);
		set_row_filter(content, "x > 1");
		PV_ASSERT_VALID(content.is_ready());

		const auto sources = content.sources();
		PV_VALID(sources.size(), size_t(1));
		const auto& [inputs, format] = sources.front();
		PV_VALID(axes_of(format), std::string("filename:string,x:number_int64,t:time"));
		PV_VALID(description_of(inputs).columns().join(",").toStdString(), std::string("x,t"));
		PV_VALID(description_of(inputs).row_filter().toStdString(), std::string("x > 1"));
		const PVRush::PVNraw& nraw = source_of(root, "json", inputs, format).get_rushnraw();
		PV_VALID(size_t(nraw.row_count()), size_t(2));
		PV_VALID(nraw.at_string(0, PVCol(1)), std::string("2"));
		PV_VALID(nraw.at_string(1, PVCol(1)), std::string("3"));

		// The source of each file has no column of the names of the files, for the condition
		// to read.
		for (QRadioButton* button : content.findChildren<QRadioButton*>()) {
			if (button->text() == "One source for each file") {
				button->setChecked(true);
			}
		}
		set_row_filter(content, "filename = 'second.ndjson'");
		PV_ASSERT_VALID(not content.is_ready());
		set_row_filter(content, "x > 1");
		const auto file_sources = content.sources();
		PV_VALID(file_sources.size(), size_t(2));
		for (const auto& [file_inputs, file_format] : file_sources) {
			PV_VALID(axes_of(file_format), std::string("x:number_int64,t:time"));
			PV_VALID(description_of(file_inputs).paths().size(), qsizetype(1));
		}
		PV_VALID(description_of(file_sources.back().first).paths().front().toStdString(),
		         second.toStdString());
		page.clear_content();
	}

	{
		// The parameters of a script, which the page does not show for.
		PVRush::PVInputType::list_inputs inputs;
		PVRush::PVFormat format;
		PVCore::PVTaskProgress progress;
		const QString params = QJsonDocument(QJsonObject{{"paths", QJsonArray{first, second}},
		                                                 {"columns", QJsonArray{"x"}},
		                                                 {"row_filter", "x < 3"}})
		                           .toJson(QJsonDocument::Compact);
		PV_ASSERT_VALID(json->create_source_description_params(params, inputs, format, progress));
		PV_VALID(axes_of(format), std::string("filename:string,x:number_int64"));
		const PVRush::PVNraw& nraw = source_of(root, "json", inputs, format).get_rushnraw();
		PV_VALID(size_t(nraw.row_count()), size_t(2));
		PV_VALID(nraw.at_string(1, PVCol(0)), std::string("second.ndjson"));

		// What is wrong with them is thrown, rather than shown.
		const QString wrong =
		    QJsonDocument(QJsonObject{{"paths", QJsonArray{first}}, {"row_filter", "nope > 1"}})
		        .toJson(QJsonDocument::Compact);
		bool thrown = false;
		try {
			inputs.clear();
			json->create_source_description_params(wrong, inputs, format, progress);
		} catch (const std::runtime_error& e) {
			thrown = std::string(e.what()).find("nope") != std::string::npos;
		}
		PV_ASSERT_VALID(thrown);
	}

	{
		// A Parquet file, of the rows of a CSV file one value of which is missing.
		const QString csv = directory.filePath("values.csv");
		write(csv, "1,,3\n4,5,6\n");
		write(csv + ".format", CSV_FORMAT);
		Squey::PVSource& csv_source = get_src_from_file(root, csv, csv + ".format");
		Squey::PVView& view =
		    csv_source.emplace_add_child().emplace_add_child().emplace_add_child();
		const QString parquet = directory.filePath("values.parquet");
		{
			PVGuiQt::PVExportSelectionDlg dialog(view);
			Q_EMIT dialog.filterSelected(".parquet files (*.parquet)");
			Squey::PVSelection selection(view.get_row_count());
			selection.select_all();
			dialog.exporter().export_rows(parquet.toStdString(), selection);
		}

		const PVRush::PVInputType_p type =
		    LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("parquet");
		PVWidgets::PVImportWidget& content = content_of(page, type, {parquet});
		PV_VALID(fill_rate_of(content, "axis1"), std::string("100 %"));
		PV_VALID(fill_rate_of(content, "axis2"), std::string("50 %"));
		item_of(content, "axis1")->setCheckState(0, Qt::Unchecked);
		set_row_filter(content, "axis1 > 1");
		const auto sources = content.sources();
		const auto& [inputs, format] = sources.front();
		PV_VALID(axes_of(format), std::string("axis2:number_uint32,axis3:number_uint32"));
		const PVRush::PVNraw& nraw = source_of(root, "parquet", inputs, format).get_rushnraw();
		PV_VALID(size_t(nraw.row_count()), size_t(1));
		PV_VALID(nraw.at_string(0, PVCol(0)), std::string("5"));
		PV_VALID(nraw.at_string(0, PVCol(1)), std::string("6"));
		page.clear_content();
	}

	return 0;
}
