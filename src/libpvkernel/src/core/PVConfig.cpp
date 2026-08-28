//
// MIT License
//
// © ESI Group, 2015
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

#include <pvkernel/core/PVConfig.h> // for PVConfig, etc
#include <pvkernel/core/PVLogger.h> // for PVLOG_ERROR
#include <pvbase/general.h> // for SQUEY_CONFDIR, etc
#include <qbytearray.h>
#include <qchar.h>
#include <qtenvironmentvariables.h>
#include <memory>    // for operator==, __shared_ptr, etc
#include <stdexcept> // for runtime_error
#include <QString>
#include <QDir>
#include <QFile>
#include <QSettings>
#include <QFileInfo>
#include <QStandardPaths>

#include <boost/dll/runtime_symbol_info.hpp>

static constexpr const char GLOBAL_CONFIG_FILENAME[] = "/opt/squey/squey.conf";

#ifdef _WIN32
	static const QString _config_dir = []() {
		QString config_dir = QString::fromLocal8Bit(qgetenv("SQUEY_CONFIG_DIR"));
		if (config_dir.isEmpty()) {
			config_dir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + QDir::separator() + SQUEY_CONFDIR;
		}
		return config_dir;
	}();
#else
	static const QString _config_dir = []() {
		QString config_dir = qgetenv("SQUEY_CONFIG_DIR");
		if (config_dir.isEmpty()) {
			config_dir = QDir::homePath() + QDir::separator() + SQUEY_CONFDIR;
		}
		return config_dir;
	}();
#endif

static const QString CONFIG_BASENAME = "/pvconfig.ini";

static const QString _lists_folder = "lists";

/*****************************************************************************
 * PVCore::PVConfig::PVConfig
 *****************************************************************************/

PVCore::PVConfig::PVConfig()
{
	QDir dir;
	dir.mkpath(_config_dir);
	dir.cd(_config_dir);

	// initialization of white/grey/black lists' directories
	dir.mkdir(get_lists_dir());
	dir.cd(get_lists_dir());
	dir.mkdir("blacklist");
	dir.mkdir("whitelist");
	dir.mkdir("greylist");

	QFileInfo fi(user_path());

	if (!fi.exists()) {
		fi.dir().mkpath(fi.path());

		QFileInfo sys_fi(shared_dir() + CONFIG_BASENAME);

		if (sys_fi.exists()) {
			QFile::copy(sys_fi.filePath(), fi.filePath());
		} else {
			PVLOG_ERROR("%s file doesn't exists\n", sys_fi.filePath().toLatin1().data());
			throw std::runtime_error("No config file found");
		}
	}

	// Compatibility with old presets file
	QSettings old_presets(QSettings::UserScope, SQUEY_ORGANISATION, SQUEY_APPLICATIONNAME);
	if (QFileInfo(old_presets.fileName()).exists()) {
		QDir().rename(old_presets.fileName(),
		              user_dir() + "/" + PRESETS_FILENAME);
	}

	_local_config = new QSettings(fi.filePath(), QSettings::IniFormat);

	if (QFileInfo(GLOBAL_CONFIG_FILENAME).exists()) {
		_global_config = new QSettings(GLOBAL_CONFIG_FILENAME, QSettings::IniFormat);
	}

#ifdef _WIN32
	_username = QString::fromLocal8Bit(qgetenv("USERNAME"));
#else
	_username = qgetenv("USER");
#endif
}

/*****************************************************************************
 * PVCore::PVConfig::~PVConfig
 *****************************************************************************/

PVCore::PVConfig::~PVConfig()
{
	if (_local_config != nullptr) {
		delete _local_config;
		_local_config = nullptr;
	}

	if (_global_config != nullptr) {
		delete _global_config;
		_global_config = nullptr;
	}
}

/*****************************************************************************
 * PVCore::PVConfig::get
 *****************************************************************************/

PVCore::PVConfig& PVCore::PVConfig::get()
{
	static PVCore::PVConfig instance;
	return instance;
}

QSettings& PVCore::PVConfig::config() const
{
	return *_local_config;
}

/*****************************************************************************
 * PVCore::PVConfig::value
 *****************************************************************************/

QVariant PVCore::PVConfig::value(const QString& name)
{
	PVCore::PVConfig& config = get();
	QVariant value;

	if (config._global_config) {
		value = config._global_config->value(name);
	}
	if (not value.isValid()) {
		value = config._local_config->value(name);
	}

	return value;
}

/*****************************************************************************
 * PVCore::PVConfig::set_value
 *****************************************************************************/

void PVCore::PVConfig::set_value(const QString& name, const QVariant& value)
{
	PVCore::PVConfig& config = get();

	config._local_config->setValue(name, value);
}

/*****************************************************************************
 * PVCore::PVConfig::get_lists_dir
 *****************************************************************************/

QString PVCore::PVConfig::get_lists_dir() const
{
	return _config_dir + QDir::separator() + _lists_folder;
}

/*****************************************************************************
 * PVCore::PVConfig::username
 *****************************************************************************/

QString PVCore::PVConfig::username()
{
	return get()._username;
}

/*****************************************************************************
 * PVCore::PVConfig::user_path
 *****************************************************************************/

QString PVCore::PVConfig::user_path()
{
	return user_dir() + CONFIG_FILENAME;
}

/*****************************************************************************
 * PVCore::PVConfig::user_dir
 *****************************************************************************/

QString PVCore::PVConfig::user_dir()
{
	return _config_dir + QDir::separator() + SQUEY_SQUEY_CONFDIR + QDir::separator();
}

/*****************************************************************************
 * PVCore::PVConfig::shared_dir
 *****************************************************************************/

QString PVCore::PVConfig::shared_dir()
{
	static const QString dir = []() -> QString {
		// An explicit answer wins over any guess: this is how a test, or a build
		// run from somewhere unusual, says where the shipped files are.
		const QString overridden = QString::fromLocal8Bit(qgetenv("SQUEY_SHARED_DIR"));
		if (not overridden.isEmpty()) {
			return overridden;
		}

		// A bundle is moved around and copied by the user; where it was built
		// says nothing about where its files are now. Only the running
		// executable does.
		const QString exe_dir =
		    QString::fromStdString(boost::dll::program_location().parent_path().string());

#if defined(__APPLE__)
		return exe_dir + "/../share/squey/squey";
#elifdef _WIN32
		return exe_dir;
#else
		if (QFileInfo::exists(QString(SQUEY_CONFIG))) {
			return QString(SQUEY_CONFIG);
		}

		// Nothing installed: a binary still in the build tree, where CMake
		// mirrors the shipped files under the layout they have once installed.
		// The executable sits at <build>/gui-qt/src/, hence the two levels up.
		return exe_dir + "/../../share/squey/squey";
#endif
	}();

	return dir;
}

/*****************************************************************************
 * PVCore::PVConfig::product_name
 *****************************************************************************/
std::string PVCore::PVConfig::product_name()
{
	return get()._product_name;
}

/*****************************************************************************
 * PVCore::PVConfig::set_product_name
 *****************************************************************************/
void PVCore::PVConfig::set_product_name(const std::string& product_name)
{
	get()._product_name = product_name;
}
