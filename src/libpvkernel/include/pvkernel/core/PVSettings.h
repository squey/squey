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

#ifndef PVCORE_PVSETTINGS_H
#define PVCORE_PVSETTINGS_H

#include <pvkernel/export.h>

#include <QHash>
#include <QJsonObject>
#include <QJsonValue>
#include <QObject>
#include <QString>
#include <QStringList>
#include <QVariant>

#include <optional>
#include <shared_mutex>
#include <utility>
#include <vector>

namespace PVCore
{

enum class PVSettingType {
	Bool,
	Int,
	Double,
	String,
	/** A directory, written as a string. */
	Path,
	/** One string among the values of PVSettingDescriptor::choices. */
	Choice
};

struct PVSettingChoice {
	QString value;
	QString label;
};

/**
 * A user setting: the values it accepts, and how the settings dialog shows it.
 *
 * The key is also the name of the setting in settings.json: dotted components in
 * snake case, all but the last naming the section the setting belongs to
 * ("listing.max_cell_size" is in "listing").
 *
 * Values are held as bool, qlonglong, double or QString, whatever the type a
 * PVSetting reads them as.
 */
struct PVSettingDescriptor {
	QString key;
	PVSettingType type = PVSettingType::String;
	QVariant default_value;
	QString title;
	QString description;
	/** Bounds of an Int or a Double setting, both included. Invalid for no bound. */
	QVariant minimum = {};
	QVariant maximum = {};
	/** Unit of a number, shown after it: "ms", "%". */
	QString unit = {};
	std::vector<PVSettingChoice> choices = {};
	/** Lengths of the shortest and of the longest String accepted, in characters. */
	int min_length = 0;
	/** Negative for no limit. */
	int max_length = -1;
	/** The setting is read when Squey starts, and changing it takes a restart. */
	bool requires_restart = false;
	/** Words that the search of the settings dialog matches besides the texts above. */
	QStringList keywords = {};
};

struct PVSettingsProblem {
	enum class Severity { Error, Warning };

	Severity severity;
	/** Empty for a problem with the document as a whole. */
	QString key;
	QString message;
};

/**
 * The descriptions of the settings, and their values.
 *
 * A setting takes the first value set, from the highest priority down:
 * - in the file of the administrator, /opt/squey/squey.conf, which the user
 *   cannot override: the setting is locked;
 * - in settings.json, the file of the user, in PVConfig::user_dir();
 * - as long as there is no settings.json, in config.ini, the configuration file
 *   of the user before it, for the keys it used to hold (pvkernel/nraw_tmp...);
 * - in pvconfig.ini, the defaults shipped with the application, in
 *   PVConfig::shared_dir();
 * - in its descriptor.
 * The two INI files of the installation accept both the keys of the settings and
 * the configuration keys that preceded them.
 *
 * A value that does not suit its setting is ignored, the next one down is used.
 *
 * The first settings.json saved takes the values of config.ini that differ from
 * the defaults. config.ini is left as it is, for the versions of Squey that
 * precede settings.json.
 *
 * Thread-safety: every member function may be called from any thread. changed()
 * is emitted from the thread that made the change, once the change is visible.
 */
class PVKERNEL_EXPORT PVSettings : public QObject
{
	Q_OBJECT

  public:
	struct files_t {
		QString user;
		QString legacy_user;
		QString shipped;
		QString administrator;
	};

	static constexpr const char* USER_FILENAME = "settings.json";

  public:
	/**
	 * The settings of the installation and of the user, as PVConfig locates them.
	 *
	 * Nothing is read before a value is first asked for, so that settings can be
	 * declared while the libraries are loaded.
	 */
	static PVSettings& get();

	/** Settings read from the files given instead: for tests. */
	explicit PVSettings(files_t files);

	~PVSettings() override;

  public:
	/**
	 * Declares a setting. A setting declared again replaces the previous
	 * declaration, and keeps its place.
	 */
	void declare(PVSettingDescriptor descriptor);

	/**
	 * Names the section of the settings whose keys start with \p key and a dot.
	 * Sections are listed in the order they are declared in.
	 */
	void declare_section(QString const& key, QString const& title);

	/** In the order of their first declaration. */
	std::vector<PVSettingDescriptor> descriptors() const;
	std::optional<PVSettingDescriptor> descriptor(QString const& key) const;

	/** Keys and titles of the declared sections, in the order of their declaration. */
	std::vector<std::pair<QString, QString>> sections() const;

	/** The declared title, or the last component of the key, spelled as words. */
	QString section_title(QString const& key) const;

	/** Invalid for a key that was not declared. */
	QVariant value(QString const& key) const;

	/** The value of the setting when the user has not set it. */
	QVariant default_value(QString const& key) const;

	/** The setting is set by the administrator, and the user cannot change it. */
	bool is_locked(QString const& key) const;

	/**
	 * Sets the value of the user, and saves it. A value equal to the default one
	 * unsets the setting instead. Writes nothing when nothing changes.
	 *
	 * \return false, with the reason in \p error, when the value does not suit
	 *         the setting, when the setting is locked, or when the file cannot be
	 *         written. The value applies even when it cannot be saved.
	 *
	 * A settings.json that cannot be read is left as it is, rather than replaced
	 * by the one setting changed: see user_file_error().
	 */
	bool set_value(QString const& key, QVariant const& value, QString* error = nullptr);

	/** Unsets the value of the user, and saves it. Same failures as set_value(). */
	bool reset(QString const& key, QString* error = nullptr);

	/**
	 * The settings of the user, as settings.json holds them once saved: keys that
	 * were not declared, and values that do not suit their setting, included.
	 */
	QJsonObject user_settings() const;

	/** What is wrong with \p settings, to be shown to the user before saving them. */
	std::vector<PVSettingsProblem> check(QJsonObject const& settings) const;

	/**
	 * Replaces the settings of the user with \p settings, and saves them as they
	 * are, keys that were not declared included. Replaces a settings.json that
	 * cannot be read too.
	 *
	 * \return false, with the reason in \p error, when the file cannot be
	 *         written. The settings apply even then.
	 */
	bool set_user_settings(QJsonObject const& settings, QString* error = nullptr);

	/** Reads the files again, for changes made outside of Squey. */
	void reload();

	QString user_file() const;
	QString administrator_file() const;

	/**
	 * Why settings.json could not be read, empty when it could. Its settings are
	 * ignored until set_user_settings() replaces it.
	 */
	QString user_file_error() const;

  public:
	/**
	 * The value \p json stands for in settings.json, or nothing, with the reason
	 * in \p error, when it does not suit the setting.
	 */
	static std::optional<QVariant> from_json(PVSettingDescriptor const& descriptor,
	                                         QJsonValue const& json,
	                                         QString* error = nullptr);

	/** How settings.json writes \p value, which has to suit the setting. */
	static QJsonValue to_json(PVSettingDescriptor const& descriptor, QVariant const& value);

  Q_SIGNALS:
	/** The value of the setting \p key changed. */
	void changed(QString const& key);

  private:
	PVSettings();

	struct entry_t {
		PVSettingDescriptor descriptor;
		QVariant value;
		QVariant default_value;
		bool locked = false;
	};

	void load_locked() const;
	void read_files_locked() const;
	void resolve_locked(entry_t& entry) const;
	std::optional<QVariant> user_value_locked(PVSettingDescriptor const& descriptor) const;
	QJsonObject user_settings_locked() const;
	/** Makes \p user the settings of the user, in place of what config.ini holds. */
	void replace_user_locked(QJsonObject const& user);
	bool save_locked(QString* error, bool replace_unreadable = false);
	std::vector<QString> resolve_all_locked();

  private:
	mutable std::shared_mutex _mutex;
	mutable std::optional<files_t> _files;
	mutable bool _loaded = false;

	mutable std::vector<entry_t> _entries;
	QHash<QString, size_t> _index;
	std::vector<std::pair<QString, QString>> _sections;

	mutable QJsonObject _user;
	mutable QString _user_file_error;
	/** Values of the INI files, by key of setting, as written there. */
	mutable QHash<QString, QString> _legacy_user;
	mutable QHash<QString, QString> _shipped;
	mutable QHash<QString, QString> _administrator;
};

/**
 * A setting declared on construction, and read with the type its users need.
 *
 * Meant to be a static object: libraries and plugins declare their settings
 * while they are loaded.
 */
template <class T>
class PVSetting
{
  public:
	explicit PVSetting(PVSettingDescriptor descriptor) : _key(descriptor.key)
	{
		PVSettings::get().declare(std::move(descriptor));
	}

	QString const& key() const { return _key; }

	T value() const { return PVSettings::get().value(_key).template value<T>(); }

	/** See PVSettings::set_value(). */
	bool set_value(T const& value, QString* error = nullptr) const
	{
		return PVSettings::get().set_value(_key, QVariant::fromValue(value), error);
	}

  private:
	QString _key;
};

/** A section declared on construction: see PVSettings::declare_section(). */
struct PVSettingsSection {
	PVSettingsSection(QString const& key, QString const& title)
	{
		PVSettings::get().declare_section(key, title);
	}
};

} // namespace PVCore

#endif // PVCORE_PVSETTINGS_H
