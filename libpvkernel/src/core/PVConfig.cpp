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

#include <pvbase/general.h> // for INENDI_CONFDIR, etc

#include <memory>    // for operator==, __shared_ptr, etc
#include <stdexcept> // for runtime_error

#include <QString>
#include <QDir>
#include <QFile>
#include <QSettings>
#include <QFileInfo>

static constexpr const char GLOBAL_CONFIG_FILENAME[] = "/opt/inendi/inspector.conf";
static constexpr const char LOCAL_CONFIG_FILENAME[] = INENDI_CONFIG "/pvconfig.ini";

PVCore::PVConfig::PVConfig_p PVCore::PVConfig::_pvconfig;

static const QString _config_dir = QDir::homePath() + QDir::separator() + INENDI_CONFDIR;
static const QString _lists_folder = "lists";

/*****************************************************************************
 * PVCore::PVConfig::PVConfig
 *****************************************************************************/

PVCore::PVConfig::PVConfig()
{
	// initialization of white/grey/black lists' directories
	QDir dir;
	dir.mkdir(_config_dir);
	dir.cd(_config_dir);
	dir.mkdir(get_lists_dir());
	dir.cd(get_lists_dir());
	dir.mkdir("blacklist");
	dir.mkdir("whitelist");
	dir.mkdir("greylist");

	QFileInfo fi(user_path());

	if (!fi.exists()) {
		fi.dir().mkpath(fi.path());

		QFileInfo sys_fi(LOCAL_CONFIG_FILENAME);

		if (sys_fi.exists()) {
			QFile::copy(sys_fi.filePath(), fi.filePath());
		} else {
			PVLOG_ERROR("%s file doesn't exists\n", fi.filePath().toLatin1().data());
			throw std::runtime_error("No config file found");
		}
	}

	// Compatibility with old presets file
	QSettings old_presets(QSettings::UserScope, INENDI_ORGANISATION, INENDI_APPLICATIONNAME);
	if (QFileInfo(old_presets.fileName()).exists()) {
		QDir().rename(old_presets.fileName(),
		              QString::fromStdString(user_dir()) + QDir::separator() + PRESETS_FILENAME);
	}

	_local_config = new QSettings(fi.filePath(), QSettings::IniFormat);

	if (QFileInfo(GLOBAL_CONFIG_FILENAME).exists()) {
		_global_config = new QSettings(GLOBAL_CONFIG_FILENAME, QSettings::IniFormat);
	}

	_username = qgetenv("USER");
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
	if (_pvconfig == nullptr) {
		_pvconfig = PVConfig_p(new PVConfig());
	}
	return *_pvconfig;
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
	return QString::fromStdString(user_dir()) + CONFIG_FILENAME;
}

/*****************************************************************************
 * PVCore::PVConfig::user_dir
 *****************************************************************************/

std::string PVCore::PVConfig::user_dir()
{
	return (QDir::homePath() + QDir::separator() + INENDI_INSPECTOR_CONFDIR + QDir::separator())
	    .toStdString();
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
