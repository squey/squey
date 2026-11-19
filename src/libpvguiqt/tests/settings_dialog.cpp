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

// The settings dialog: what a search lists, how an edit reaches the draft and
// then settings.json, and the JSON mode, which has to agree with the list.
//
// Runs under "-platform offscreen", with a configuration directory of its own
// (see CMakeLists.txt).

#include <pvguiqt/PVCodeEditorStyle.h>
#include <pvguiqt/PVGuiQtSettings.h>
#include <pvguiqt/PVSettingsDialog.h>

#include <pvkernel/core/PVConfig.h>
#include <pvkernel/core/PVKernelSettings.h>
#include <pvkernel/core/PVSettings.h>
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/widgets/PVCSVExporterWidget.h>

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QDialogButtonBox>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QListWidget>
#include <QPushButton>
#include <QSpinBox>
#include <QTest>
#include <QTextEdit>
#include <QToolButton>

#include <string>

namespace
{

QWidget* row(PVGuiQt::PVSettingsDialog& dialog, QString const& key)
{
	auto* row = dialog.findChild<QWidget*>("setting:" + key);
	PV_ASSERT_VALID(row != nullptr, "key", key.toStdString());
	return row;
}

template <class Editor>
Editor* editor(PVGuiQt::PVSettingsDialog& dialog, QString const& key)
{
	auto* editor = dialog.findChild<Editor*>("editor:" + key);
	PV_ASSERT_VALID(editor != nullptr, "key", key.toStdString());
	return editor;
}

bool listed(PVGuiQt::PVSettingsDialog& dialog, QString const& key)
{
	return not row(dialog, key)->isHidden();
}

std::string compact(QJsonObject const& object)
{
	return QJsonDocument(object).toJson(QJsonDocument::Compact).toStdString();
}

std::string file_content(QString const& path)
{
	QFile file(path);
	return file.open(QIODevice::ReadOnly) ? file.readAll().toStdString() : std::string();
}

void write(QString const& path, QByteArray const& content)
{
	QFile file(path);
	PV_ASSERT_VALID(file.open(QIODevice::WriteOnly | QIODevice::Truncate), "file",
	                path.toStdString());
	file.write(content);
}

void search_lists_the_settings_that_match(PVGuiQt::PVSettingsDialog& dialog)
{
	for (PVCore::PVSettingDescriptor const& descriptor : PVCore::PVSettings::get().descriptors()) {
		PV_ASSERT_VALID(listed(dialog, descriptor.key), "key", descriptor.key.toStdString());
	}

	// Every word has to match, whatever its case.
	dialog.set_filter("CHUNK size");
	PV_ASSERT_VALID(listed(dialog, "import.chunk_size"));
	PV_ASSERT_VALID(not listed(dialog, "import.parallel_chunks"));
	PV_ASSERT_VALID(not listed(dialog, "appearance.color_scheme"));

	// The title of the section, the keys and the values of the choices match too.
	dialog.set_filter("import");
	PV_ASSERT_VALID(listed(dialog, "import.parallel_chunks"));
	dialog.set_filter("appearance.color_scheme");
	PV_ASSERT_VALID(listed(dialog, "appearance.color_scheme"));
	dialog.set_filter("light");
	PV_ASSERT_VALID(listed(dialog, "appearance.color_scheme"));

	dialog.set_filter("@modified");
	PV_ASSERT_VALID(not listed(dialog, "import.chunk_size"));
	PV_ASSERT_VALID(not dialog.findChild<QLabel*>("no_match")->isHidden());
	dialog.set_filter("");
}

void edits_are_saved_by_apply(PVGuiQt::PVSettingsDialog& dialog, QString const& user_file)
{
	auto* chunk_size = editor<QSpinBox>(dialog, "import.chunk_size");
	chunk_size->setValue(20000);
	PV_VALID(dialog.draft()["import.chunk_size"].toInteger(), qint64(20000));
	// Nothing is saved before Apply.
	PV_ASSERT_VALID(not QFile::exists(user_file));
	PV_VALID(PVCore::settings::import_chunk_size.value(), 10048576);

	dialog.set_filter("@modified");
	PV_ASSERT_VALID(listed(dialog, "import.chunk_size"));
	PV_ASSERT_VALID(not listed(dialog, "appearance.color_scheme"));
	dialog.set_filter("");

	PV_ASSERT_VALID(dialog.apply());
	PV_VALID(PVCore::settings::import_chunk_size.value(), 20000);
	PV_VALID(file_content(user_file).empty(), false);
	PV_VALID(
	    compact(
	        QJsonDocument::fromJson(QByteArray::fromStdString(file_content(user_file))).object()),
	    std::string(R"({"import.chunk_size":20000})"));

	// Back to the default value, the setting leaves the draft.
	chunk_size->setValue(10048576);
	PV_ASSERT_VALID(not dialog.draft().contains("import.chunk_size"));
	chunk_size->setValue(20000);

	// Typed key by key, a number keeps what was typed before.
	auto* font_size = editor<QDoubleSpinBox>(dialog, "editor.font_size");
	font_size->selectAll();
	QTest::keyClicks(font_size, QString("12") + font_size->locale().decimalPoint() + "5");
	PV_VALID(font_size->value(), 12.5);
	PV_VALID(dialog.draft()["editor.font_size"].toDouble(), 12.5);
	font_size->setValue(10.5);
}

void json_mode_edits_the_same_draft(PVGuiQt::PVSettingsDialog& dialog)
{
	dialog.set_json_mode(true);
	PV_ASSERT_VALID(dialog.json_mode());
	auto* user_json = dialog.findChild<QTextEdit*>("user_json");
	PV_ASSERT_VALID(user_json->toPlainText().contains("\"import.chunk_size\": 20000"));
	PV_ASSERT_VALID(dialog.findChild<QTextEdit*>("defaults_json")
	                    ->toPlainText()
	                    .contains("\"import.chunk_size\": 10048576"));

	user_json->setPlainText(
	    R"({"import.chunk_size": 30000, "unknown.key": true, "export.csv_header": "yes"})");
	// The problems are listed once the typing settles.
	QTest::qWait(500);
	auto* problems = dialog.findChild<QListWidget*>("problems");
	PV_VALID(problems->count(), 2);

	dialog.set_json_mode(false);
	PV_ASSERT_VALID(not dialog.json_mode());
	PV_VALID(editor<QSpinBox>(dialog, "import.chunk_size")->value(), 30000);
	// What the list cannot show stays in the draft, and the list says what it ignores.
	PV_ASSERT_VALID(dialog.draft().contains("unknown.key"));
	auto* notes = row(dialog, "export.csv_header")->findChild<QLabel*>("notes");
	PV_ASSERT_VALID(not notes->isHidden() and notes->text().contains("ignored"), "notes",
	                notes->text().toStdString());

	// Text that does not parse keeps the dialog on the JSON.
	dialog.set_json_mode(true);
	user_json->setPlainText("{");
	dialog.set_json_mode(false);
	PV_ASSERT_VALID(dialog.json_mode());
	PV_ASSERT_VALID(not dialog.apply());
	user_json->setPlainText(R"({"import.chunk_size": 30000, "export.csv_header": false})");
	dialog.set_json_mode(false);
	PV_ASSERT_VALID(not dialog.json_mode());
	PV_VALID(editor<QCheckBox>(dialog, "export.csv_header")->isChecked(), false);
}

void resets_reach_the_draft(PVGuiQt::PVSettingsDialog& dialog)
{
	// The button resets what the search lists, and only that.
	dialog.set_filter("chunk size");
	dialog.findChild<QDialogButtonBox*>()->button(QDialogButtonBox::RestoreDefaults)->click();
	PV_ASSERT_VALID(not dialog.draft().contains("import.chunk_size"));
	PV_ASSERT_VALID(dialog.draft().contains("export.csv_header"));
	dialog.set_filter("");

	row(dialog, "export.csv_header")->findChild<QToolButton*>("reset")->click();
	PV_ASSERT_VALID(not dialog.draft().contains("export.csv_header"));
	PV_VALID(editor<QCheckBox>(dialog, "export.csv_header")->isChecked(), true);
}

} // namespace

int main(int argc, char** argv)
{
	QApplication app(argc, argv); // argv carries "-platform offscreen"

	// SQUEY_CONFIG_DIR gives the test a configuration of its own.
	const QString user_file = PVCore::PVConfig::user_dir() + PVCore::PVSettings::USER_FILENAME;
	QFile::remove(user_file);

	{
		PVGuiQt::PVSettingsDialog dialog;
		dialog.show();
		QApplication::processEvents();

		search_lists_the_settings_that_match(dialog);
		edits_are_saved_by_apply(dialog, user_file);
		json_mode_edits_the_same_draft(dialog);
		resets_reach_the_draft(dialog);

		// Cancel throws the draft away.
		editor<QSpinBox>(dialog, "import.chunk_size")->setValue(40000);
		dialog.reject();
		PV_VALID(PVCore::settings::import_chunk_size.value(), 20000);
	}

	// A change made elsewhere shows when the dialog opens again.
	PV_ASSERT_VALID(PVCore::settings::lines_platform.set_value("qt"));
	{
		PVGuiQt::PVSettingsDialog dialog;
		auto* platforms = editor<QComboBox>(dialog, "rendering.lines_platform");
		PV_VALID(platforms->currentData().toString().toStdString(), std::string("qt"));
		// The machine adds its own platforms, if any, to those every machine offers.
		PV_ASSERT_VALID(platforms->findData("auto") == 0);
		PV_ASSERT_VALID(platforms->findData("cpu") < 0);
	}

	// A platform this machine does not offer is listed all the same, as such.
	PV_ASSERT_VALID(PVCore::settings::lines_platform.set_value("gpu:Absent platform"));
	{
		PVGuiQt::PVSettingsDialog dialog;
		auto* platforms = editor<QComboBox>(dialog, "rendering.lines_platform");
		PV_VALID(platforms->currentData().toString().toStdString(),
		         std::string("gpu:Absent platform"));
		PV_VALID(platforms->currentText().toStdString(),
		         std::string("gpu:Absent platform, not found on this machine"));
	}
	PV_ASSERT_VALID(PVCore::PVSettings::get().reset(PVCore::settings::lines_platform.key()));

	// The code editors follow their font, those already open included.
	{
		QTextEdit code;
		PVGuiQt::style_code_editor(&code, "JSON");
		PV_ASSERT_VALID(PVGuiQt::settings::editor_font_size.set_value(14.0));
		PV_VALID(code.document()->defaultFont().pointSizeF(), 14.0);
	}

	// The CSV export starts with the options it was last left with.
	PV_ASSERT_VALID(PVCore::settings::export_csv_separator.set_value(";"));
	PV_ASSERT_VALID(PVCore::settings::export_csv_header.set_value(false));
	{
		PVWidgets::PVCSVExporterWidget exporter_widget;
		PV_VALID(exporter_widget.exporter().get_sep_char(), std::string(";"));
		PV_VALID(exporter_widget.exporter().get_quote_char(), std::string("\""));
		PV_VALID(exporter_widget.exporter().get_export_header(), false);
	}

	// An unreadable settings.json opens as it is, in the JSON mode, so that it
	// can be fixed rather than lost.
	write(user_file, "{ broken");
	{
		PVGuiQt::PVSettingsDialog dialog;
		PV_ASSERT_VALID(dialog.json_mode());
		auto* user_json = dialog.findChild<QTextEdit*>("user_json");
		PV_VALID(user_json->toPlainText().toStdString(), std::string("{ broken"));
		user_json->setPlainText(R"({"import.chunk_size": 50000})");
		PV_ASSERT_VALID(dialog.apply());
		PV_VALID(PVCore::settings::import_chunk_size.value(), 50000);
	}

	QFile::remove(user_file);
	return 0;
}
