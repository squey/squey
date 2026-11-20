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

// The import of text files, as the import page asks for it: the files, whether they make one
// source or one each, and their format, the one next to them, a file the user chooses or names
// on the command line, one the user makes in the format builder of the page, or else the one
// Squey detects from their lines, which the format keeps in memory rather than next to the data:
// their splitter, the types of their columns, and names from a header, which it skips.

#include "common.h"
#include "test-env.h"

#include <pvguiqt/PVImportPage.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVTaskProgress.h>
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/widgets/PVImportWidget.h>

#include <QApplication>
#include <QFile>
#include <QPushButton>
#include <QRadioButton>
#include <QTemporaryDir>
#include <QTest>

#include <memory>
#include <string>

namespace
{

constexpr const char* CSV_FORMAT = R"(<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE PVParamXml>
<param version="11" first_line="0">
 <splitter type="csv" sep="," quote="&quot;">
  <field><axis name="first" type="number_uint32"/></field>
  <field><axis name="second" type="number_uint32"/></field>
 </splitter>
</param>
)";

void write(QString const& path, QByteArray const& content)
{
	QFile file(path);
	PV_ASSERT_VALID(file.open(QIODevice::WriteOnly), "file", path.toStdString());
	file.write(content);
}

/** The full paths of the formats of the sources of @p content, in their order. */
std::string format_paths(PVWidgets::PVImportWidget const& content)
{
	std::string paths;
	for (const auto& [inputs, format] : content.sources()) {
		paths += (paths.empty() ? "" : ";") + format.get_full_path().toStdString();
	}
	return paths;
}

} // namespace

int main(int argc, char** argv)
{
	init_env();
	QApplication app(argc, argv); // argv carries "-platform offscreen"

	QTemporaryDir directory;
	PV_ASSERT_VALID(directory.isValid());
	const QString with_format = directory.filePath("with.csv");
	const QString plain = directory.filePath("plain.csv");
	const QString named = directory.filePath("named.format");
	write(with_format, "1,2\n3,4\n");
	write(with_format + ".format", CSV_FORMAT);
	write(named, CSV_FORMAT);
	QByteArray rows = "name,count,ratio\n";
	for (int row = 0; row < 50; row++) {
		rows += "host" + QByteArray::number(row) + "," + QByteArray::number(row) + "," +
		        QByteArray::number(row + 0.5) + "\n";
	}
	write(plain, rows);

	const PVRush::PVInputType_p type =
	    LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("file");
	PV_ASSERT_VALID(type->has_generated_format());
	const std::unique_ptr<PVWidgets::PVImportWidget> content(type->create_import_widget(nullptr));
	PV_ASSERT_VALID(not content->is_ready());

	// Formats and investigations chosen along with the files are none of them.
	content->set_paths({with_format, with_format + ".format", directory.filePath("old.pvi")});
	PV_ASSERT_VALID(content->is_ready());
	PV_VALID(content->sources().front().first.size(), qsizetype(1));
	PV_VALID(format_paths(*content), (with_format + ".format").toStdString());

	// One format found for the files of a source reads them all, else the import detects it.
	content->set_paths({with_format, plain});
	PV_VALID(content->sources().size(), size_t(1));
	PV_VALID(content->sources().front().first.size(), qsizetype(2));
	PV_VALID(format_paths(*content), (with_format + ".format").toStdString());
	content->findChild<QRadioButton*>("import_source_per_file")->setChecked(true);
	PV_VALID(format_paths(*content), (with_format + ".format;").toStdString());
	PV_VALID(content->sources().back().second.get_axes().size(), qsizetype(0));

	// A format the user chooses, or a command line names, reads all of the files.
	content->set_format(named);
	PV_VALID(format_paths(*content), (named + ";" + named).toStdString());
	content->set_format(directory.filePath("missing.format"));
	bool refused = false;
	try {
		content->sources();
	} catch (const std::runtime_error&) {
		refused = true;
	}
	PV_ASSERT_VALID(refused);

	// The detection, out of the GUI thread, of the file with no format.
	PVCore::PVTaskProgress progress;
	const PVRush::PVFormat detected = type->get_format_from_inputs(
	    {PVRush::PVInputDescription_p(new PVRush::PVFileDescription(plain, false))}, &progress);
	PV_VALID(detected.get_full_path().toStdString(), std::string());
	const QList<PVRush::PVAxisFormat>& axes = detected.get_axes();
	PV_VALID(axes.size(), qsizetype(3));
	PV_VALID(axes[0].get_name().toStdString(), std::string("name"));
	PV_VALID(axes[1].get_name().toStdString(), std::string("count"));
	PV_VALID(axes[0].get_type().toStdString(), std::string("string"));
	PV_ASSERT_VALID(axes[1].get_type().startsWith("number_"), "type",
	                axes[1].get_type().toStdString());
	PV_ASSERT_VALID(axes[2].get_type().startsWith("number_"), "type",
	                axes[2].get_type().toStdString());
	PV_VALID(detected.get_first_line(), size_t(1));
	PV_ASSERT_VALID(not QFile::exists(plain + ".format"));

	// The format builder of the page, in place of the content, makes the format from the one
	// detected, without asking anything.
	PVGuiQt::PVImportPage page;
	page.show();
	PVWidgets::PVImportWidget* edited = type->create_import_widget(nullptr);
	page.set_content(edited, "Import: Text", {}, -1);
	edited->set_paths({plain});
	auto* edit = edited->findChild<QPushButton*>("import_format_edit");
	auto* made = edited->findChild<QRadioButton*>("import_format_made");
	PV_ASSERT_VALID(not made->isVisible());
	edit->click();
	PV_ASSERT_VALID(QTest::qWaitFor([&]() { return page.editing_format(); }, 60000));
	PV_ASSERT_VALID(not edited->isVisible());
	PV_ASSERT_VALID(QApplication::activeModalWidget() == nullptr);
	page.findChild<QPushButton*>("import_format_use")->click();
	PV_ASSERT_VALID(not page.editing_format() and edited->isVisible());
	PV_ASSERT_VALID(made->isVisible() and made->isChecked());
	PV_ASSERT_VALID(not edited->findChild<QRadioButton*>("import_format_detected")->isChecked());
	const PVRush::PVFormat made_format = edited->sources().front().second;
	PV_VALID(made_format.get_axes().size(), qsizetype(3));
	PV_VALID(made_format.get_axes()[1].get_name().toStdString(), std::string("count"));
	PV_VALID(made_format.get_full_path().toStdString(), std::string());
	PV_ASSERT_VALID(not QFile::exists(plain + ".format"));

	// Canceled, an edit keeps the format made before.
	edit->click();
	PV_ASSERT_VALID(page.editing_format());
	page.findChild<QPushButton*>("import_format_cancel")->click();
	PV_ASSERT_VALID(not page.editing_format() and made->isChecked());

	return 0;
}
