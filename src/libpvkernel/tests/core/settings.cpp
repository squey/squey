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

#include <pvkernel/core/PVArgumentSettings.h>
#include <pvkernel/core/PVConfig.h>
#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/core/PVPlainTextType.h>
#include <pvkernel/core/PVSettings.h>
#include <pvkernel/core/squey_assert.h>

#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QSettings>
#include <QStringList>
#include <QTemporaryDir>

#include <algorithm>
#include <string>

namespace PVAssert
{
template <>
void printer<QJsonObject>(const char* expr, const QJsonObject& value, const QJsonObject& expected)
{
	std::cerr << expr << " = " << QJsonDocument(value).toJson(QJsonDocument::Compact).toStdString()
	          << " (expected: "
	          << QJsonDocument(expected).toJson(QJsonDocument::Compact).toStdString() << ")"
	          << std::endl;
}

template <>
void printer<QString>(const char* expr, const QString& value, const QString& expected)
{
	std::cerr << expr << " = " << value.toStdString() << " (expected: " << expected.toStdString()
	          << ")" << std::endl;
}
} // namespace PVAssert

using PVCore::PVSettings;
using PVCore::PVSettingType;

namespace
{

PVSettings::files_t files_in(QTemporaryDir const& dir)
{
	return {dir.filePath("settings.json"), dir.filePath("config.ini"), dir.filePath("pvconfig.ini"),
	        dir.filePath("squey.conf")};
}

void write(QString const& path, QByteArray const& content)
{
	QFile file(path);
	PV_ASSERT_VALID(file.open(QIODevice::WriteOnly | QIODevice::Truncate), "file",
	                path.toStdString());
	file.write(content);
}

QByteArray read(QString const& path)
{
	QFile file(path);
	PV_ASSERT_VALID(file.open(QIODevice::ReadOnly), "file", path.toStdString());
	return file.readAll();
}

QJsonObject read_json(QString const& path)
{
	return QJsonDocument::fromJson(read(path)).object();
}

std::string text(QVariant const& value)
{
	return value.toString().toStdString();
}

void declare_all(PVSettings& settings)
{
	settings.declare({.key = "test.flag",
	                  .type = PVSettingType::Bool,
	                  .default_value = false,
	                  .title = "Flag",
	                  .description = "Flag of the test"});
	settings.declare({.key = "test.count",
	                  .type = PVSettingType::Int,
	                  .default_value = 10,
	                  .title = "Count",
	                  .description = "Count of the test",
	                  .minimum = 1,
	                  .maximum = 100});
	settings.declare({.key = "test.ratio",
	                  .type = PVSettingType::Double,
	                  .default_value = 0.5,
	                  .title = "Ratio",
	                  .description = "Ratio of the test",
	                  .minimum = 0.0,
	                  .maximum = 1.0});
	settings.declare({.key = "test.name",
	                  .type = PVSettingType::String,
	                  .default_value = "squey",
	                  .title = "Name",
	                  .description = "Name of the test",
	                  .min_length = 1,
	                  .max_length = 8});
	settings.declare({.key = "test.choice",
	                  .type = PVSettingType::Choice,
	                  .default_value = "b",
	                  .title = "Choice",
	                  .description = "Choice of the test",
	                  .choices = {{"a", "A"}, {"b", "B"}}});
	// Settings that replace configuration keys of config.ini.
	settings.declare({.key = "import.chunk_size",
	                  .type = PVSettingType::Int,
	                  .default_value = 409600,
	                  .title = "Chunk size",
	                  .description = "Chunk size of the test",
	                  .minimum = 4096});
	settings.declare({.key = "general.working_directory",
	                  .type = PVSettingType::Path,
	                  .default_value = "/tmp/squey",
	                  .title = "Working directory",
	                  .description = "Working directory of the test"});
	// Whose configuration key, a boolean, became one of its choices.
	settings.declare({.key = "rendering.lines_platform",
	                  .type = PVSettingType::Choice,
	                  .default_value = "auto",
	                  .title = "Lines platform",
	                  .description = "Lines platform of the test",
	                  .choices = {{"auto", "Automatic"}, {"cpu", "Processor"}},
	                  .machine_choices = [] {
		                  return std::vector<PVCore::PVSettingChoice>{{"auto", "Automatic"},
		                                                              {"gpu:Here", "Here"}};
	                  }});
}

bool has_problem(std::vector<PVCore::PVSettingsProblem> const& problems,
                 QString const& key,
                 PVCore::PVSettingsProblem::Severity severity)
{
	return std::any_of(problems.begin(), problems.end(), [&](auto const& problem) {
		return problem.key == key and problem.severity == severity;
	});
}

void defaults_apply_without_files()
{
	QTemporaryDir dir;
	PVSettings settings(files_in(dir));
	declare_all(settings);

	PV_VALID(settings.value("test.count").toLongLong(), 10LL);
	PV_VALID(settings.value("test.flag").toBool(), false);
	PV_VALID(text(settings.value("test.name")), std::string("squey"));
	PV_VALID(text(settings.value("test.choice")), std::string("b"));
	PV_ASSERT_VALID(not settings.value("test.undeclared").isValid());
	PV_ASSERT_VALID(settings.user_settings().isEmpty());
	PV_ASSERT_VALID(not settings.is_locked("test.count"));
	// Neither reading the settings nor setting them as they are writes them.
	PV_ASSERT_VALID(settings.set_value("test.count", 10));
	PV_ASSERT_VALID(settings.reset("test.flag"));
	PV_ASSERT_VALID(not QFile::exists(dir.filePath("settings.json")));
}

void user_values_are_checked()
{
	QTemporaryDir dir;
	write(dir.filePath("settings.json"),
	      R"({"test.count": 42, "test.flag": true, "test.choice": "z", "test.ratio": 2,
	          "other.key": [1, 2]})");
	PVSettings settings(files_in(dir));
	declare_all(settings);

	PV_VALID(settings.value("test.count").toLongLong(), 42LL);
	PV_VALID(settings.value("test.flag").toBool(), true);
	// Values that do not suit their setting are ignored.
	PV_VALID(text(settings.value("test.choice")), std::string("b"));
	PV_VALID(settings.value("test.ratio").toDouble(), 0.5);

	const auto problems = settings.check(settings.user_settings());
	PV_VALID(problems.size(), size_t(3));
	PV_ASSERT_VALID(
	    has_problem(problems, "test.choice", PVCore::PVSettingsProblem::Severity::Error));
	PV_ASSERT_VALID(
	    has_problem(problems, "test.ratio", PVCore::PVSettingsProblem::Severity::Error));
	PV_ASSERT_VALID(
	    has_problem(problems, "other.key", PVCore::PVSettingsProblem::Severity::Warning));

	QString error;
	PV_ASSERT_VALID(not settings.set_value("test.count", 1000, &error));
	PV_ASSERT_VALID(not error.isEmpty());
	PV_ASSERT_VALID(not settings.set_value("test.count", "12"));
	PV_ASSERT_VALID(not settings.set_value("test.count", 1.5));
	PV_ASSERT_VALID(not settings.set_value("test.name", "too long a name"));
	PV_ASSERT_VALID(not settings.set_value("test.name", ""));
	PV_ASSERT_VALID(not settings.set_value("test.choice", "c"));
	PV_ASSERT_VALID(not settings.set_value("test.undeclared", 1));
	PV_VALID(settings.value("test.count").toLongLong(), 42LL);

	const bool saved_name = settings.set_value("test.name", "pv", &error);
	PV_ASSERT_VALID(saved_name, "error", error.toStdString());
	QJsonObject saved = read_json(dir.filePath("settings.json"));
	PV_VALID(text(saved["test.name"].toVariant()), std::string("pv"));
	// What the user wrote is kept, even what Squey does not understand.
	PV_VALID(text(saved["test.choice"].toVariant()), std::string("z"));
	PV_ASSERT_VALID(saved["other.key"].isArray());

	// The default value unsets the setting.
	PV_ASSERT_VALID(settings.set_value("test.count", 10));
	saved = read_json(dir.filePath("settings.json"));
	PV_ASSERT_VALID(not saved.contains("test.count"));
	PV_VALID(settings.value("test.count").toLongLong(), 10LL);
}

void machine_choices_accept_other_values()
{
	QTemporaryDir dir;
	// As the settings of another machine may be.
	write(dir.filePath("settings.json"), R"({"rendering.lines_platform": "gpu:Elsewhere"})");
	PVSettings settings(files_in(dir));
	declare_all(settings);

	PV_VALID(text(settings.value("rendering.lines_platform")), std::string("gpu:Elsewhere"));
	PV_ASSERT_VALID(settings.check(settings.user_settings()).empty());
	PV_ASSERT_VALID(settings.set_value("rendering.lines_platform", "cpu:Elsewhere"));
	PV_ASSERT_VALID(not settings.set_value("rendering.lines_platform", 3));
	PV_VALID(text(settings.value("rendering.lines_platform")), std::string("cpu:Elsewhere"));
}

void changes_are_signalled()
{
	QTemporaryDir dir;
	PVSettings settings(files_in(dir));
	declare_all(settings);
	QStringList changed;
	QObject::connect(&settings, &PVSettings::changed,
	                 [&changed](QString const& key) { changed << key; });

	PV_ASSERT_VALID(settings.set_value("test.count", 20));
	PV_VALID(changed.join(",").toStdString(), std::string("test.count"));
	PV_ASSERT_VALID(settings.set_value("test.count", 20));
	PV_VALID(changed.size(), qsizetype(1));

	PV_ASSERT_VALID(settings.set_user_settings(QJsonObject{{"test.flag", true}}));
	changed.sort();
	PV_VALID(changed.join(",").toStdString(), std::string("test.count,test.count,test.flag"));
	PV_VALID(read_json(dir.filePath("settings.json")), (QJsonObject{{"test.flag", true}}));

	changed.clear();
	PV_ASSERT_VALID(settings.reset("test.flag"));
	PV_VALID(changed.join(",").toStdString(), std::string("test.flag"));
}

void shipped_and_legacy_values_are_read()
{
	QTemporaryDir dir;
	const QByteArray shipped = "[pvkernel]\n"
	                           "max_size_chunk=10048576\n"
	                           "nraw_tmp=/srv/tmp-squey\n"
	                           "\n"
	                           "[backend_opencl]\n"
	                           "force_cpu=false\n"
	                           "\n"
	                           "[General]\n"
	                           "test.count=7\n";
	write(dir.filePath("pvconfig.ini"), shipped);
	// config.ini starts as a copy of pvconfig.ini, which the user then edits.
	write(dir.filePath("config.ini"), "[pvkernel]\n"
	                                  "max_size_chunk=10048576\n"
	                                  "nraw_tmp=/data/squey\n"
	                                  "\n"
	                                  "[backend_opencl]\n"
	                                  "force_cpu=true\n"
	                                  "\n"
	                                  "[duckdb]\n"
	                                  "memory_percent=30\n"
	                                  "\n"
	                                  "[squey]\n"
	                                  "sample_dataset_offered=true\n");
	PVSettings settings(files_in(dir));
	declare_all(settings);

	PV_VALID(settings.default_value("import.chunk_size").toLongLong(), 10048576LL);
	PV_VALID(settings.value("import.chunk_size").toLongLong(), 10048576LL);
	PV_VALID(settings.default_value("test.count").toLongLong(), 7LL);
	PV_VALID(text(settings.default_value("general.working_directory")),
	         std::string("/srv/tmp-squey"));
	PV_VALID(text(settings.value("general.working_directory")), std::string("/data/squey"));
	// A boolean configuration key turned into a choice sets it when true only.
	PV_VALID(text(settings.default_value("rendering.lines_platform")), std::string("auto"));
	PV_VALID(text(settings.value("rendering.lines_platform")), std::string("cpu"));

	// The values the user did not change are not theirs. Those of settings this
	// process does not declare are kept all the same.
	PV_VALID(settings.user_settings(), (QJsonObject{{"general.working_directory", "/data/squey"},
	                                                {"rendering.lines_platform", "cpu"},
	                                                {"sql.memory_percent", 30}}));

	// Saving takes them to settings.json, and leaves config.ini to the versions
	// of Squey that read it.
	const QByteArray config = read(dir.filePath("config.ini"));
	PV_ASSERT_VALID(settings.set_value("test.flag", true));
	PV_VALID(read_json(dir.filePath("settings.json")),
	         (QJsonObject{{"general.working_directory", "/data/squey"},
	                      {"rendering.lines_platform", "cpu"},
	                      {"sql.memory_percent", 30},
	                      {"test.flag", true}}));
	PV_VALID(read(dir.filePath("config.ini")).toStdString(), config.toStdString());

	// Unset, a setting takes the shipped default again, not the old value of
	// config.ini.
	PV_ASSERT_VALID(settings.reset("general.working_directory"));
	PV_VALID(text(settings.value("general.working_directory")), std::string("/srv/tmp-squey"));
	PVSettings reread(files_in(dir));
	declare_all(reread);
	PV_VALID(text(reread.value("general.working_directory")), std::string("/srv/tmp-squey"));
	PV_VALID(text(reread.value("rendering.lines_platform")), std::string("cpu"));
}

void legacy_values_can_be_unset_first()
{
	QTemporaryDir dir;
	write(dir.filePath("config.ini"), "[backend_opencl]\n"
	                                  "force_cpu=true\n");
	PVSettings settings(files_in(dir));
	declare_all(settings);
	PV_VALID(text(settings.value("rendering.lines_platform")), std::string("cpu"));

	// The first change of all is the one that unsets the value of config.ini.
	PV_ASSERT_VALID(settings.set_value("rendering.lines_platform", "auto"));
	PV_VALID(text(settings.value("rendering.lines_platform")), std::string("auto"));
	PV_VALID(read_json(dir.filePath("settings.json")), QJsonObject());
	PVSettings reread(files_in(dir));
	declare_all(reread);
	PV_VALID(text(reread.value("rendering.lines_platform")), std::string("auto"));
}

void administrator_values_win()
{
	QTemporaryDir dir;
	write(dir.filePath("squey.conf"), "[General]\n"
	                                  "test.count=3\n"
	                                  "\n"
	                                  "[backend_opencl]\n"
	                                  "force_cpu=true\n");
	write(dir.filePath("settings.json"), R"({"test.count": 50, "test.flag": true})");
	PVSettings settings(files_in(dir));
	declare_all(settings);

	PV_VALID(settings.value("test.count").toLongLong(), 3LL);
	PV_ASSERT_VALID(settings.is_locked("test.count"));
	PV_VALID(text(settings.value("rendering.lines_platform")), std::string("cpu"));
	PV_ASSERT_VALID(settings.is_locked("rendering.lines_platform"));
	PV_VALID(settings.value("test.flag").toBool(), true);
	PV_ASSERT_VALID(not settings.is_locked("test.flag"));

	PV_ASSERT_VALID(has_problem(settings.check(settings.user_settings()), "test.count",
	                            PVCore::PVSettingsProblem::Severity::Warning));
	PV_ASSERT_VALID(not settings.set_value("test.count", 4));
	PV_ASSERT_VALID(not settings.reset("test.count"));
	PV_VALID(settings.value("test.count").toLongLong(), 3LL);
}

void unreadable_user_file_is_kept()
{
	QTemporaryDir dir;
	const QByteArray broken = R"({"test.count": 42,)";
	write(dir.filePath("settings.json"), broken);
	PVSettings settings(files_in(dir));
	declare_all(settings);

	PV_VALID(settings.value("test.count").toLongLong(), 10LL);
	PV_ASSERT_VALID(not settings.user_file_error().isEmpty());

	// One setting changed elsewhere must not replace every other the user wrote.
	QString error;
	PV_ASSERT_VALID(not settings.set_value("test.flag", true, &error));
	PV_ASSERT_VALID(not error.isEmpty());
	PV_VALID(settings.value("test.flag").toBool(), true);
	PV_VALID(read(dir.filePath("settings.json")).toStdString(), broken.toStdString());

	// Replacing the whole file is the way to fix it.
	PV_ASSERT_VALID(settings.set_user_settings(QJsonObject{{"test.count", 43}}));
	PV_ASSERT_VALID(settings.user_file_error().isEmpty());
	PV_VALID(settings.value("test.count").toLongLong(), 43LL);
	PV_VALID(read_json(dir.filePath("settings.json")), (QJsonObject{{"test.count", 43}}));
}

void reload_reads_outside_changes()
{
	QTemporaryDir dir;
	PVSettings settings(files_in(dir));
	declare_all(settings);
	PV_VALID(settings.value("test.count").toLongLong(), 10LL);

	QStringList changed;
	QObject::connect(&settings, &PVSettings::changed,
	                 [&changed](QString const& key) { changed << key; });
	write(dir.filePath("settings.json"), R"({"test.count": 5})");
	settings.reload();
	PV_VALID(settings.value("test.count").toLongLong(), 5LL);
	PV_VALID(changed.join(",").toStdString(), std::string("test.count"));
}

void late_declarations_are_resolved()
{
	QTemporaryDir dir;
	write(dir.filePath("settings.json"), R"({"plugin.option": "x"})");
	PVSettings settings(files_in(dir));
	declare_all(settings);
	PV_VALID(settings.value("test.count").toLongLong(), 10LL);

	// As a plugin does, once the settings were read.
	settings.declare({.key = "plugin.option",
	                  .type = PVSettingType::Choice,
	                  .default_value = "y",
	                  .title = "Option",
	                  .description = "Option of the test",
	                  .choices = {{"x", "X"}, {"y", "Y"}}});
	PV_VALID(text(settings.value("plugin.option")), std::string("x"));
	PV_VALID(settings.section_title("plugin"), QString("Plugin"));
	settings.declare_section({.key = "plugin", .title = "Some plugin"});
	PV_VALID(settings.section_title("plugin"), QString("Some plugin"));
}

void typed_settings_use_the_files_of_the_user()
{
	// SQUEY_CONFIG_DIR points the configuration of the test at a directory of its own.
	QFile::remove(PVCore::PVConfig::user_dir() + PVSettings::USER_FILENAME);

	static const PVCore::PVSetting<int> count({.key = "test.typed",
	                                           .type = PVSettingType::Int,
	                                           .default_value = 3,
	                                           .title = "Typed",
	                                           .description = "Typed of the test"});
	PV_VALID(count.value(), 3);
	PV_ASSERT_VALID(count.set_value(4));
	PV_VALID(count.value(), 4);
	PV_VALID(read_json(PVCore::PVConfig::user_dir() + PVSettings::USER_FILENAME),
	         (QJsonObject{{"test.typed", 4}}));
	QFile::remove(PVCore::PVConfig::user_dir() + PVSettings::USER_FILENAME);
}

void argument_settings_are_generated()
{
	QFile::remove(PVCore::PVConfig::user_dir() + PVSettings::USER_FILENAME);

	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey("flag", "A flag")] = true;
	args[PVCore::PVArgumentKey("count", "A count")] = 3;
	args[PVCore::PVArgumentKey("name", "A name")] = QString("squey");
	args[PVCore::PVArgumentKey("sep", "A separator")] = QChar(',');
	args[PVCore::PVArgumentKey("mode", "A mode")].setValue(
	    PVCore::PVEnumType(QStringList{"first", "second"}, 0));
	args[PVCore::PVArgumentKey("text", "A text")].setValue(PVCore::PVPlainTextType("x"));

	const PVCore::PVArgumentKeyList declared = PVCore::declare_argument_settings(
	    "test.plugin", args, args.keys(), "Default of the test plugin");
	// Free text is no setting.
	PV_VALID(declared.size(), size_t(5));
	PVSettings& settings = PVSettings::get();
	const auto mode = settings.descriptor("test.plugin.mode");
	PV_ASSERT_VALID(mode.has_value());
	PV_ASSERT_VALID(mode->type == PVSettingType::Choice);
	PV_VALID(mode->title, QString("A mode"));
	PV_VALID(text(mode->default_value), std::string("first"));
	PV_ASSERT_VALID(not settings.descriptor("test.plugin.text").has_value());

	// Without a choice of the user, the arguments are left as they are.
	PVCore::PVArgumentList applied = args;
	PVCore::apply_argument_settings("test.plugin", applied);
	PV_VALID(applied["mode"].value<PVCore::PVEnumType>().get_sel_index(), 0);
	PV_VALID(int(applied["sep"].toChar().unicode()), int(','));

	PV_ASSERT_VALID(settings.set_value("test.plugin.mode", "second"));
	PV_ASSERT_VALID(settings.set_value("test.plugin.sep", ";"));
	PV_ASSERT_VALID(settings.set_value("test.plugin.count", 7));
	PV_ASSERT_VALID(settings.set_value("test.plugin.flag", false));
	PV_ASSERT_VALID(not settings.set_value("test.plugin.sep", ";;"));
	PVCore::apply_argument_settings("test.plugin", applied);
	PV_VALID(applied["mode"].value<PVCore::PVEnumType>().get_sel_index(), 1);
	PV_VALID(int(applied["sep"].toChar().unicode()), int(';'));
	PV_VALID(applied["count"].toInt(), 7);
	PV_VALID(applied["count"].typeId(), int(QMetaType::Int));
	PV_VALID(applied["flag"].toBool(), false);
	PV_VALID(applied["text"].value<PVCore::PVPlainTextType>().get_text().toStdString(),
	         std::string("x"));
	QFile::remove(PVCore::PVConfig::user_dir() + PVSettings::USER_FILENAME);
}

} // namespace

int main()
{
	defaults_apply_without_files();
	user_values_are_checked();
	machine_choices_accept_other_values();
	changes_are_signalled();
	shipped_and_legacy_values_are_read();
	legacy_values_can_be_unset_first();
	administrator_values_win();
	unreadable_user_file_is_kept();
	reload_reads_outside_changes();
	late_declarations_are_resolved();
	typed_settings_use_the_files_of_the_user();
	argument_settings_are_generated();
	return 0;
}
