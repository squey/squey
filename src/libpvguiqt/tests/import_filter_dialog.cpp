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

// The choice of the columns and the rows to import, which the imports of JSON and Parquet files
// offer before they read them: the columns found, with their types and the share of the rows
// that give them a value, a condition refused while it is wrong, and the choice made, which the
// import keeps. The parameters of a script make the same choice.

#include "common.h"
#include "test-env.h"

#include <pvguiqt/PVExportSelectionDlg.h>

#include <squey/PVRoot.h>
#include <squey/PVScene.h>
#include <squey/PVSelection.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVFileListDescription.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/widgets/PVImportFilterDialog.h>

#include <QApplication>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QPlainTextEdit>
#include <QTemporaryDir>
#include <QTimer>
#include <QTreeWidget>

#include <functional>
#include <string>

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

using choice_f = std::function<void(PVWidgets::PVImportFilterDialog&)>;

void write(QString const& path, QByteArray const& content)
{
	QFile file(path);
	PV_ASSERT_VALID(file.open(QIODevice::WriteOnly), "file", path.toStdString());
	file.write(content);
}

/**
 * Has @p choose make the choice of the user, once the dialog of the columns and rows to import
 * shows.
 */
void when_asked(const choice_f& choose)
{
	QTimer::singleShot(10, [choose] {
		auto* dialog =
		    qobject_cast<PVWidgets::PVImportFilterDialog*>(QApplication::activeModalWidget());
		if (dialog == nullptr) {
			when_asked(choose);
			return;
		}
		choose(*dialog);
	});
}

QTreeWidgetItem* item_of(PVWidgets::PVImportFilterDialog& dialog, QString const& column)
{
	const QList<QTreeWidgetItem*> items =
	    dialog.findChild<QTreeWidget*>()->findItems(column, Qt::MatchExactly);
	PV_VALID(items.size(), qsizetype(1), "column", column.toStdString());
	return items.front();
}

std::string fill_rate_of(PVWidgets::PVImportFilterDialog& dialog, QString const& column)
{
	return item_of(dialog, column)->text(2).toStdString();
}

void uncheck(PVWidgets::PVImportFilterDialog& dialog, QString const& column)
{
	item_of(dialog, column)->setCheckState(0, Qt::Unchecked);
}

void set_row_filter(PVWidgets::PVImportFilterDialog& dialog, QString const& condition)
{
	dialog.findChild<QPlainTextEdit*>()->setPlainText(condition);
}

/** The inputs and the format that @p type makes of @p paths, once @p choose made the choice. */
std::pair<PVRush::PVInputType::list_inputs, PVRush::PVFormat>
import_files(const PVRush::PVInputType_p& type, QStringList const& paths, const choice_f& choose)
{
	when_asked(choose);
	PVRush::hash_formats formats;
	PVRush::PVInputType::list_inputs inputs;
	QString format_name;
	PVCore::PVArgumentList arguments;
	PV_ASSERT_VALID(type->create_widget_with_input_files(paths, formats, inputs, format_name,
	                                                     arguments, nullptr));
	return {inputs, formats[format_name]};
}

Squey::PVSource& source_of(Squey::PVRoot& root,
                           QString const& type,
                           PVRush::PVInputType::list_inputs const& inputs,
                           PVRush::PVFormat const& format)
{
	Squey::PVSource& source =
	    root.emplace_add_child(type.toStdString())
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

	{
		const auto [inputs, format] =
		    import_files(json, {first, second}, [](PVWidgets::PVImportFilterDialog& dialog) {
			    PV_VALID(fill_rate_of(dialog, "x"), std::string("100 %"));
			    PV_VALID(fill_rate_of(dialog, "y.z"), std::string("67 %"));
			    PV_VALID(fill_rate_of(dialog, "t"), std::string("33 %"));
			    uncheck(dialog, "y.z");

			    set_row_filter(dialog, "nope > 1");
			    dialog.accept();
			    PV_ASSERT_VALID(dialog.isVisible());
			    bool told = false;
			    for (const QLabel* label : dialog.findChildren<QLabel*>()) {
				    told |= label->isVisible() and label->text().contains("nope");
			    }
			    PV_ASSERT_VALID(told);

			    set_row_filter(dialog, "x > 1");
			    dialog.accept();
		    });
		PV_VALID(axes_of(format), std::string("filename:string,x:number_int64,t:time"));
		PV_VALID(description_of(inputs).columns().join(",").toStdString(), std::string("x,t"));
		PV_VALID(description_of(inputs).row_filter().toStdString(), std::string("x > 1"));
		const PVRush::PVNraw& nraw = source_of(root, "json", inputs, format).get_rushnraw();
		PV_VALID(size_t(nraw.row_count()), size_t(2));
		PV_VALID(nraw.at_string(0, PVCol(1)), std::string("2"));
		PV_VALID(nraw.at_string(1, PVCol(1)), std::string("3"));
	}

	{
		// The parameters of a script, which the dialog does not show for.
		PVRush::PVInputType::list_inputs inputs;
		PVRush::PVFormat format;
		const QString params = QJsonDocument(QJsonObject{{"paths", QJsonArray{first, second}},
		                                                 {"columns", QJsonArray{"x"}},
		                                                 {"row_filter", "x < 3"}})
		                           .toJson(QJsonDocument::Compact);
		PV_ASSERT_VALID(json->create_source_description_params(params, inputs, format));
		PV_VALID(axes_of(format), std::string("filename:string,x:number_int64"));
		const PVRush::PVNraw& nraw = source_of(root, "json", inputs, format).get_rushnraw();
		PV_VALID(size_t(nraw.row_count()), size_t(2));
		PV_VALID(nraw.at_string(1, PVCol(0)), std::string("second.ndjson"));
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
		const auto [inputs, format] =
		    import_files(type, {parquet}, [](PVWidgets::PVImportFilterDialog& dialog) {
			    PV_VALID(fill_rate_of(dialog, "axis1"), std::string("100 %"));
			    PV_VALID(fill_rate_of(dialog, "axis2"), std::string("50 %"));
			    uncheck(dialog, "axis1");
			    set_row_filter(dialog, "axis1 > 1");
			    dialog.accept();
		    });
		PV_VALID(axes_of(format), std::string("axis2:number_uint32,axis3:number_uint32"));
		const PVRush::PVNraw& nraw = source_of(root, "parquet", inputs, format).get_rushnraw();
		PV_VALID(size_t(nraw.row_count()), size_t(1));
		PV_VALID(nraw.at_string(0, PVCol(0)), std::string("5"));
		PV_VALID(nraw.at_string(0, PVCol(1)), std::string("6"));
	}

	return 0;
}
