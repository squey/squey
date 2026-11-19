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

// The choice of the axes an export writes: all of the axes, those of the combination of the view,
// or those of a combination edited for the export, which starts as that of the view.

#include "common.h"
#include "test-env.h"

#include <pvguiqt/PVAxesCombinationExportWidget.h>
#include <pvguiqt/PVCSVExporterWidget.h>

#include <squey/PVRoot.h>
#include <squey/PVSelection.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvkernel/core/squey_assert.h>

#include <QApplication>
#include <QFile>
#include <QPushButton>
#include <QRadioButton>
#include <QTemporaryDir>

#include <string>

namespace
{

using Choice = PVGuiQt::PVAxesCombinationExportWidget::Choice;

constexpr const char* FORMAT = R"(<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE PVParamXml>
<param version="11" first_line="0">
 <splitter type="csv" sep="," quote="&quot;">
  <field>
   <axis name="axis1" type="number_uint32">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
  <field>
   <axis name="axis2" type="number_uint32">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
  <field>
   <axis name="axis3" type="number_uint32">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
 </splitter>
</param>
)";

void write(QString const& path, QByteArray const& content)
{
	QFile file(path);
	PV_ASSERT_VALID(file.open(QIODevice::WriteOnly), "file", path.toStdString());
	file.write(content);
}

QRadioButton* radio(QWidget& widget, QString const& text)
{
	for (auto* button : widget.findChildren<QRadioButton*>()) {
		if (button->text() == text) {
			return button;
		}
	}
	PV_ASSERT_VALID(false, "no radio button", text.toStdString());
	return nullptr;
}

std::string columns_of(PVGuiQt::PVAxesCombinationExportWidget const& widget)
{
	std::string columns;
	for (PVCol column : widget.column_indexes()) {
		columns += (columns.empty() ? "" : ",") + std::to_string(int(column));
	}
	return columns;
}

/** The first line of the CSV file @p widget exports @p view into, in @p directory. */
std::string exported_header(PVGuiQt::PVCSVExporterWidget& widget,
                            Squey::PVView const& view,
                            QTemporaryDir const& directory)
{
	Squey::PVSelection selection(view.get_row_count());
	selection.select_all();
	const QString path = directory.filePath("exported.csv");
	widget.exporter().export_rows(path.toStdString(), selection);

	QFile file(path);
	PV_ASSERT_VALID(file.open(QIODevice::ReadOnly));
	return file.readLine().trimmed().toStdString();
}

} // namespace

int main(int argc, char** argv)
{
	init_env();

	QTemporaryDir inputs;
	PV_ASSERT_VALID(inputs.isValid());
	const QString file = inputs.filePath("axes.csv");
	write(file, "1,2,3\n4,5,6\n");
	write(file + ".format", FORMAT);

	Squey::PVRoot root;
	Squey::PVSource& src = get_src_from_file(root, file, file + ".format");
	Squey::PVView& view = src.emplace_add_child().emplace_add_child().emplace_add_child();
	view.set_axes_combination({PVCol(2), PVCol(0)});

	QApplication app(argc, argv); // argv carries "-platform offscreen"

	{
		PVGuiQt::PVAxesCombinationExportWidget axes(view, Choice::Current, "All of them");
		int changes = 0;
		QObject::connect(&axes, &PVGuiQt::PVAxesCombinationExportWidget::changed,
		                 [&changes]() { changes++; });
		auto* edit = axes.findChild<QPushButton*>();

		PV_VALID(columns_of(axes), std::string("2,0"));
		PV_VALID(axes.names().join(",").toStdString(), std::string("axis3,axis1"));
		PV_VALID(edit->isEnabled(), false);

		radio(axes, "All of them")->click();
		PV_ASSERT_VALID(axes.choice() == Choice::All);
		PV_VALID(columns_of(axes), std::string("0,1,2"));
		PV_VALID(axes.names().join(",").toStdString(), std::string("axis1,axis2,axis3"));
		PV_VALID(changes, 1);

		// Edited for the export only, the combination starts as that of the view.
		radio(axes, "Use custom axes combination")->click();
		PV_ASSERT_VALID(axes.choice() == Choice::Custom);
		PV_VALID(columns_of(axes), std::string("2,0"));
		PV_VALID(edit->isEnabled(), true);
		PV_VALID(changes, 2);
	}

	{
		QTemporaryDir directory;
		PV_ASSERT_VALID(directory.isValid());
		PVGuiQt::PVCSVExporterWidget csv(view);
		PV_VALID(exported_header(csv, view, directory), std::string("#axis3,axis1"));

		radio(csv, "Use all axes combination")->click();
		PV_VALID(exported_header(csv, view, directory), std::string("#axis1,axis2,axis3"));
	}

	return 0;
}
