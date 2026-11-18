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

#include <pvkernel/core/PVSettings.h>

#include <pvkernel/core/PVConfig.h>

#include <pvlogger.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QSaveFile>
#include <QSettings>

#include <cmath>
#include <mutex>

namespace
{

/**
 * The configuration keys that settings replaced, and the settings that replace
 * them.
 */
struct legacy_key_t {
	const char* legacy;
	const char* key;
	/** For a process that does not declare the setting, and saves its value all the same. */
	PVCore::PVSettingType type;
};

constexpr legacy_key_t LEGACY_KEYS[] = {
    {"gui/theme_scheme", "appearance.color_scheme", PVCore::PVSettingType::String},
    {"pvkernel/nraw_tmp", "general.working_directory", PVCore::PVSettingType::Path},
    {"pvkernel/max_size_chunk", "import.chunk_size", PVCore::PVSettingType::Int},
    {"pvkernel/number_living_chunks", "import.parallel_chunks", PVCore::PVSettingType::Int},
    {"backend_opencl/force_cpu", "rendering.force_cpu", PVCore::PVSettingType::Bool},
    {"backend_opencl/platform_index", "rendering.opencl_platform", PVCore::PVSettingType::Int},
    {"duckdb/memory_percent", "sql.memory_percent", PVCore::PVSettingType::Int},
    {"duckdb/max_temp_percent", "sql.spill_percent", PVCore::PVSettingType::Int},
    {"scatter_thumbnails/thumbnail_size", "scatter_thumbnails.size", PVCore::PVSettingType::Int},
    {"scatter_thumbnails/sort_by_correlation", "scatter_thumbnails.sort_by_correlation",
     PVCore::PVSettingType::Bool},
};

/**
 * The values of an INI file, by key of setting.
 *
 * \param legacy_only Ignore the keys of settings, which a file that predates them
 *        cannot hold.
 */
QHash<QString, QString> read_ini(QString const& path, bool legacy_only)
{
	QHash<QString, QString> values;
	if (not QFileInfo::exists(path)) {
		return values;
	}

	QSettings ini(path, QSettings::IniFormat);
	auto raw = [&ini](QString const& key) {
		// QSettings splits a value that holds a comma.
		const QVariant value = ini.value(key);
		return value.typeId() == QMetaType::QStringList ? value.toStringList().join(",")
		                                                : value.toString();
	};
	for (legacy_key_t const& legacy : LEGACY_KEYS) {
		if (ini.contains(legacy.legacy)) {
			values[legacy.key] = raw(legacy.legacy);
		}
	}
	if (not legacy_only) {
		// The keys of settings win over the configuration keys they replace.
		for (QString const& key : ini.allKeys()) {
			if (not key.contains('/')) {
				values[key] = raw(key);
			}
		}
	}
	return values;
}

/**
 * The value an INI file means for the setting. Booleans are read as config.ini
 * used to read them, through QVariant.
 */
std::optional<QVariant> from_ini(PVCore::PVSettingDescriptor const& descriptor,
                                 QString const& raw,
                                 QString* error = nullptr)
{
	switch (descriptor.type) {
	case PVCore::PVSettingType::Bool:
		return PVCore::PVSettings::from_json(descriptor, QVariant(raw).toBool(), error);
	case PVCore::PVSettingType::Int: {
		bool ok = false;
		const qlonglong integer = raw.trimmed().toLongLong(&ok);
		if (not ok) {
			if (error) {
				*error = QString("\"%1\" is not an integer").arg(raw);
			}
			return std::nullopt;
		}
		return PVCore::PVSettings::from_json(descriptor, QJsonValue(integer), error);
	}
	case PVCore::PVSettingType::Double: {
		bool ok = false;
		const double number = raw.trimmed().toDouble(&ok);
		if (not ok) {
			if (error) {
				*error = QString("\"%1\" is not a number").arg(raw);
			}
			return std::nullopt;
		}
		return PVCore::PVSettings::from_json(descriptor, QJsonValue(number), error);
	}
	case PVCore::PVSettingType::String:
	case PVCore::PVSettingType::Path:
	case PVCore::PVSettingType::Choice:
		return PVCore::PVSettings::from_json(descriptor, QJsonValue(raw), error);
	}
	return std::nullopt;
}

/**
 * The sections of the settings dialog, before any library declares its settings
 * in a section of its own.
 */
constexpr std::pair<const char*, const char*> TABLE_OF_CONTENTS[] = {
    {"appearance", "Appearance"},
    {"general", "General"},
    {"import", "Import"},
    {"export", "Export"},
    {"listing", "Listing"},
    {"statistics", "Statistics"},
    {"parallel_view", "Parallel coordinates"},
    {"scatter_thumbnails", "Scatter plots gallery"},
    {"sql", "SQL console"},
    {"editor", "Code editors"},
    {"rendering", "Rendering"},
    {"plugins", "Plugins"},
};

QString words(QString const& component)
{
	QString title = component;
	title.replace('_', ' ').replace('-', ' ');
	if (not title.isEmpty()) {
		title[0] = title[0].toUpper();
	}
	return title;
}

QString bound_text(QVariant const& bound)
{
	return bound.typeId() == QMetaType::Double ? QString::number(bound.toDouble())
	                                           : QString::number(bound.toLongLong());
}

} // namespace

PVCore::PVSettings& PVCore::PVSettings::get()
{
	static PVSettings instance;
	return instance;
}

PVCore::PVSettings::PVSettings()
{
	for (auto const& [key, title] : TABLE_OF_CONTENTS) {
		_sections.emplace_back(key, title);
	}
}

PVCore::PVSettings::PVSettings(files_t files) : _files(std::move(files)) {}

PVCore::PVSettings::~PVSettings() = default;

void PVCore::PVSettings::declare(PVSettingDescriptor descriptor)
{
	// Held in the type values are read as, so that comparing them compares values.
	if (auto default_value = from_json(descriptor, to_json(descriptor, descriptor.default_value))) {
		descriptor.default_value = *default_value;
	} else {
		pvlogger::error() << "Setting " << qPrintable(descriptor.key)
		                  << " has a default value that does not suit it" << std::endl;
	}

	std::unique_lock lock(_mutex);
	auto it = _index.find(descriptor.key);
	if (it == _index.end()) {
		_index.insert(descriptor.key, _entries.size());
		_entries.push_back({std::move(descriptor), {}, {}, false});
		it = _index.find(_entries.back().descriptor.key);
	} else {
		_entries[*it].descriptor = std::move(descriptor);
	}
	if (_loaded) {
		resolve_locked(_entries[*it]);
	}
}

void PVCore::PVSettings::declare_section(QString const& key, QString const& title)
{
	std::unique_lock lock(_mutex);
	for (auto& section : _sections) {
		if (section.first == key) {
			section.second = title;
			return;
		}
	}
	_sections.emplace_back(key, title);
}

std::vector<PVCore::PVSettingDescriptor> PVCore::PVSettings::descriptors() const
{
	std::shared_lock lock(_mutex);
	std::vector<PVSettingDescriptor> descriptors;
	descriptors.reserve(_entries.size());
	for (entry_t const& entry : _entries) {
		descriptors.push_back(entry.descriptor);
	}
	return descriptors;
}

std::optional<PVCore::PVSettingDescriptor> PVCore::PVSettings::descriptor(QString const& key) const
{
	std::shared_lock lock(_mutex);
	auto it = _index.find(key);
	if (it == _index.end()) {
		return std::nullopt;
	}
	return _entries[*it].descriptor;
}

std::vector<std::pair<QString, QString>> PVCore::PVSettings::sections() const
{
	std::shared_lock lock(_mutex);
	return _sections;
}

QString PVCore::PVSettings::section_title(QString const& key) const
{
	std::shared_lock lock(_mutex);
	for (auto const& [section_key, title] : _sections) {
		if (section_key == key) {
			return title;
		}
	}
	return words(key.section('.', -1));
}

QVariant PVCore::PVSettings::value(QString const& key) const
{
	auto lookup = [this, &key]() {
		auto it = _index.find(key);
		return it == _index.end() ? QVariant() : _entries[*it].value;
	};
	{
		std::shared_lock lock(_mutex);
		if (_loaded) {
			return lookup();
		}
	}
	std::unique_lock lock(_mutex);
	load_locked();
	return lookup();
}

QVariant PVCore::PVSettings::default_value(QString const& key) const
{
	std::unique_lock lock(_mutex);
	load_locked();
	auto it = _index.find(key);
	return it == _index.end() ? QVariant() : _entries[*it].default_value;
}

bool PVCore::PVSettings::is_locked(QString const& key) const
{
	std::unique_lock lock(_mutex);
	load_locked();
	auto it = _index.find(key);
	return it != _index.end() and _entries[*it].locked;
}

bool PVCore::PVSettings::set_value(QString const& key, QVariant const& value, QString* error)
{
	std::vector<QString> changed_keys;
	bool saved = false;
	{
		std::unique_lock lock(_mutex);
		load_locked();
		auto it = _index.find(key);
		if (it == _index.end()) {
			if (error) {
				*error = QString("There is no setting %1").arg(key);
			}
			return false;
		}
		entry_t const& entry = _entries[*it];
		if (entry.locked) {
			if (error) {
				*error = QString("%1 is set by the administrator, in %2")
				             .arg(key, _files->administrator);
			}
			return false;
		}
		const std::optional<QVariant> checked =
		    from_json(entry.descriptor, QJsonValue::fromVariant(value), error);
		if (not checked) {
			return false;
		}

		const QJsonObject before = user_settings_locked();
		QJsonObject user = before;
		if (*checked == entry.default_value) {
			user.remove(key);
		} else {
			user[key] = to_json(entry.descriptor, *checked);
		}
		if (user == before) {
			return true;
		}
		replace_user_locked(user);
		changed_keys = resolve_all_locked();
		saved = save_locked(error);
	}
	for (QString const& changed_key : changed_keys) {
		Q_EMIT changed(changed_key);
	}
	return saved;
}

bool PVCore::PVSettings::reset(QString const& key, QString* error)
{
	std::vector<QString> changed_keys;
	bool saved = false;
	{
		std::unique_lock lock(_mutex);
		load_locked();
		auto it = _index.find(key);
		if (it != _index.end() and _entries[*it].locked) {
			if (error) {
				*error = QString("%1 is set by the administrator, in %2")
				             .arg(key, _files->administrator);
			}
			return false;
		}
		QJsonObject user = user_settings_locked();
		if (not user.contains(key)) {
			return true;
		}
		user.remove(key);
		replace_user_locked(user);
		changed_keys = resolve_all_locked();
		saved = save_locked(error);
	}
	for (QString const& changed_key : changed_keys) {
		Q_EMIT changed(changed_key);
	}
	return saved;
}

QJsonObject PVCore::PVSettings::user_settings() const
{
	std::unique_lock lock(_mutex);
	load_locked();
	return user_settings_locked();
}

std::vector<PVCore::PVSettingsProblem> PVCore::PVSettings::check(QJsonObject const& settings) const
{
	std::unique_lock lock(_mutex);
	load_locked();
	std::vector<PVSettingsProblem> problems;
	for (auto it = settings.begin(); it != settings.end(); ++it) {
		auto entry = _index.find(it.key());
		if (entry == _index.end()) {
			problems.push_back(
			    {PVSettingsProblem::Severity::Warning, it.key(),
			     "Unknown setting: misspelled, or from a plugin that is not loaded"});
			continue;
		}
		QString error;
		if (not from_json(_entries[*entry].descriptor, it.value(), &error)) {
			problems.push_back({PVSettingsProblem::Severity::Error, it.key(), error});
		} else if (_entries[*entry].locked) {
			problems.push_back(
			    {PVSettingsProblem::Severity::Warning, it.key(),
			     QString("Ignored: the administrator sets it, in %1").arg(_files->administrator)});
		}
	}
	return problems;
}

bool PVCore::PVSettings::set_user_settings(QJsonObject const& settings, QString* error)
{
	std::vector<QString> changed_keys;
	bool saved = false;
	{
		std::unique_lock lock(_mutex);
		load_locked();
		replace_user_locked(settings);
		changed_keys = resolve_all_locked();
		saved = save_locked(error, true);
	}
	for (QString const& changed_key : changed_keys) {
		Q_EMIT changed(changed_key);
	}
	return saved;
}

void PVCore::PVSettings::reload()
{
	std::vector<QString> changed_keys;
	{
		std::unique_lock lock(_mutex);
		if (not _loaded) {
			return;
		}
		read_files_locked();
		changed_keys = resolve_all_locked();
	}
	for (QString const& changed_key : changed_keys) {
		Q_EMIT changed(changed_key);
	}
}

QString PVCore::PVSettings::user_file() const
{
	std::unique_lock lock(_mutex);
	load_locked();
	return _files->user;
}

QString PVCore::PVSettings::administrator_file() const
{
	std::unique_lock lock(_mutex);
	load_locked();
	return _files->administrator;
}

QString PVCore::PVSettings::user_file_error() const
{
	std::unique_lock lock(_mutex);
	load_locked();
	return _user_file_error;
}

std::optional<QVariant> PVCore::PVSettings::from_json(PVSettingDescriptor const& descriptor,
                                                      QJsonValue const& json,
                                                      QString* error)
{
	auto fail = [error](QString const& message) -> std::optional<QVariant> {
		if (error) {
			*error = message;
		}
		return std::nullopt;
	};
	auto out_of_bounds = [&descriptor](double number) {
		return (descriptor.minimum.isValid() and number < descriptor.minimum.toDouble()) or
		       (descriptor.maximum.isValid() and number > descriptor.maximum.toDouble());
	};
	auto bounds = [&descriptor]() {
		if (descriptor.minimum.isValid() and descriptor.maximum.isValid()) {
			return QString("between %1 and %2")
			    .arg(bound_text(descriptor.minimum), bound_text(descriptor.maximum));
		}
		if (descriptor.minimum.isValid()) {
			return QString("%1 or more").arg(bound_text(descriptor.minimum));
		}
		return QString("%1 or less").arg(bound_text(descriptor.maximum));
	};

	switch (descriptor.type) {
	case PVSettingType::Bool:
		if (not json.isBool()) {
			return fail("Expected true or false");
		}
		return QVariant(json.toBool());
	case PVSettingType::Int: {
		// Beyond 2^53, a double no longer tells an integer from its neighbours.
		constexpr double EXACT_INTEGERS = 9007199254740992.0;
		const double number = json.toDouble();
		if (not json.isDouble() or number != std::trunc(number) or
		    std::abs(number) > EXACT_INTEGERS) {
			return fail("Expected an integer");
		}
		if (out_of_bounds(number)) {
			return fail("Expected an integer " + bounds());
		}
		return QVariant(static_cast<qlonglong>(number));
	}
	case PVSettingType::Double:
		if (not json.isDouble()) {
			return fail("Expected a number");
		}
		if (out_of_bounds(json.toDouble())) {
			return fail("Expected a number " + bounds());
		}
		return QVariant(json.toDouble());
	case PVSettingType::String:
	case PVSettingType::Path:
		if (not json.isString()) {
			return fail("Expected a string");
		}
		if (json.toString().size() < descriptor.min_length) {
			return fail(QString("Expected at least %1 characters").arg(descriptor.min_length));
		}
		if (descriptor.max_length >= 0 and json.toString().size() > descriptor.max_length) {
			return fail(QString("Expected at most %1 characters").arg(descriptor.max_length));
		}
		return QVariant(json.toString());
	case PVSettingType::Choice: {
		QStringList values;
		for (PVSettingChoice const& choice : descriptor.choices) {
			if (json.isString() and json.toString() == choice.value) {
				return QVariant(choice.value);
			}
			values << QString("\"%1\"").arg(choice.value);
		}
		return fail("Expected one of " + values.join(", "));
	}
	}
	return fail("Unknown type of setting");
}

QJsonValue PVCore::PVSettings::to_json(PVSettingDescriptor const& descriptor, QVariant const& value)
{
	switch (descriptor.type) {
	case PVSettingType::Bool:
		return value.toBool();
	case PVSettingType::Int:
		return value.toLongLong();
	case PVSettingType::Double:
		return value.toDouble();
	case PVSettingType::String:
	case PVSettingType::Path:
	case PVSettingType::Choice:
		return value.toString();
	}
	return {};
}

void PVCore::PVSettings::load_locked() const
{
	if (_loaded) {
		return;
	}
	if (not _files) {
		const QString user_dir = PVConfig::user_dir();
		_files = files_t{user_dir + USER_FILENAME, user_dir + PVConfig::CONFIG_FILENAME,
		                 PVConfig::shared_dir() + "/" + PVConfig::SHIPPED_CONFIG_FILENAME,
		                 PVConfig::GLOBAL_CONFIG_PATH};
	}
	read_files_locked();
	for (entry_t& entry : _entries) {
		resolve_locked(entry);
	}
	_loaded = true;
}

void PVCore::PVSettings::read_files_locked() const
{
	_user = QJsonObject();
	_user_file_error.clear();
	QFile file(_files->user);
	if (file.exists()) {
		if (not file.open(QIODevice::ReadOnly)) {
			_user_file_error = file.errorString();
		} else {
			QJsonParseError parse_error;
			const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parse_error);
			if (parse_error.error != QJsonParseError::NoError) {
				_user_file_error = QString("%1, at character %2")
				                       .arg(parse_error.errorString())
				                       .arg(parse_error.offset);
			} else if (not document.isObject()) {
				_user_file_error = "The settings are not a JSON object";
			} else {
				_user = document.object();
			}
		}
		if (not _user_file_error.isEmpty()) {
			pvlogger::warn() << "Ignoring " << qPrintable(_files->user) << ": "
			                 << qPrintable(_user_file_error) << std::endl;
		}
	}
	// The values of config.ini are in settings.json once it exists.
	_legacy_user = file.exists() ? QHash<QString, QString>() : read_ini(_files->legacy_user, true);
	_shipped = read_ini(_files->shipped, false);
	_administrator = read_ini(_files->administrator, false);
}

void PVCore::PVSettings::resolve_locked(entry_t& entry) const
{
	PVSettingDescriptor const& descriptor = entry.descriptor;
	auto from_file = [&descriptor](QHash<QString, QString> const& values,
	                               QString const& path) -> std::optional<QVariant> {
		auto it = values.find(descriptor.key);
		if (it == values.end()) {
			return std::nullopt;
		}
		QString error;
		std::optional<QVariant> value = from_ini(descriptor, *it, &error);
		if (not value) {
			pvlogger::warn() << "Ignoring " << qPrintable(descriptor.key) << " in "
			                 << qPrintable(path) << ": " << qPrintable(error) << std::endl;
		}
		return value;
	};

	entry.default_value = from_file(_shipped, _files->shipped).value_or(descriptor.default_value);
	if (auto value = from_file(_administrator, _files->administrator)) {
		entry.value = *value;
		entry.locked = true;
		return;
	}
	entry.locked = false;
	entry.value = user_value_locked(descriptor).value_or(entry.default_value);
}

std::optional<QVariant>
PVCore::PVSettings::user_value_locked(PVSettingDescriptor const& descriptor) const
{
	if (_user.contains(descriptor.key)) {
		QString error;
		std::optional<QVariant> value = from_json(descriptor, _user[descriptor.key], &error);
		if (value) {
			return value;
		}
		pvlogger::warn() << "Ignoring " << qPrintable(descriptor.key) << " in "
		                 << qPrintable(_files->user) << ": " << qPrintable(error) << std::endl;
	}
	auto it = _legacy_user.find(descriptor.key);
	if (it != _legacy_user.end()) {
		return from_ini(descriptor, *it);
	}
	return std::nullopt;
}

QJsonObject PVCore::PVSettings::user_settings_locked() const
{
	QJsonObject settings = _user;
	for (legacy_key_t const& legacy : LEGACY_KEYS) {
		auto it = _legacy_user.find(legacy.key);
		if (settings.contains(legacy.key) or it == _legacy_user.end()) {
			continue;
		}
		// config.ini starts as a copy of pvconfig.ini: the values it still shares
		// with the defaults are not choices of the user.
		auto entry = _index.find(legacy.key);
		std::optional<QVariant> value;
		QVariant default_value;
		if (entry != _index.end()) {
			value = from_ini(_entries[*entry].descriptor, *it);
			default_value = _entries[*entry].default_value;
		} else {
			const PVSettingDescriptor undeclared{.key = legacy.key,
			                                     .type = legacy.type,
			                                     .default_value = {},
			                                     .title = {},
			                                     .description = {}};
			value = from_ini(undeclared, *it);
			auto shipped = _shipped.find(legacy.key);
			if (shipped != _shipped.end()) {
				default_value = from_ini(undeclared, *shipped).value_or(QVariant());
			}
		}
		if (value and *value != default_value) {
			settings[legacy.key] = QJsonValue::fromVariant(*value);
		}
	}
	return settings;
}

void PVCore::PVSettings::replace_user_locked(QJsonObject const& user)
{
	_user = user;
	// What config.ini holds is in the settings of the user now, or was unset.
	_legacy_user.clear();
}

bool PVCore::PVSettings::save_locked(QString* error, bool replace_unreadable)
{
	if (not _user_file_error.isEmpty() and not replace_unreadable) {
		if (error) {
			*error = QString("%1 was left as it is, since it cannot be read: %2")
			             .arg(_files->user, _user_file_error);
		}
		return false;
	}
	QDir().mkpath(QFileInfo(_files->user).path());
	QSaveFile file(_files->user);
	if (not file.open(QIODevice::WriteOnly) or
	    file.write(QJsonDocument(_user).toJson(QJsonDocument::Indented)) < 0 or not file.commit()) {
		if (error) {
			*error = QString("Cannot write %1: %2").arg(_files->user, file.errorString());
		}
		return false;
	}
	_user_file_error.clear();
	return true;
}

std::vector<QString> PVCore::PVSettings::resolve_all_locked()
{
	std::vector<QString> changed_keys;
	for (entry_t& entry : _entries) {
		const QVariant previous = entry.value;
		resolve_locked(entry);
		if (entry.value != previous) {
			changed_keys.push_back(entry.descriptor.key);
		}
	}
	return changed_keys;
}
