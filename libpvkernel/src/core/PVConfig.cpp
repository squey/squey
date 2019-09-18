/**
 * @file
 *
 * @copyright (C) Picviz Labs 1024-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

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

constexpr const char* GLOBAL_CONFIG_FILENAME = INENDI_CONFIG "/pvconfig.ini";

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

		QFileInfo sys_fi(GLOBAL_CONFIG_FILENAME);

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

	_config = new QSettings(fi.filePath(), QSettings::IniFormat);

	_username = qgetenv("USER");
}

/*****************************************************************************
 * PVCore::PVConfig::~PVConfig
 *****************************************************************************/

PVCore::PVConfig::~PVConfig()
{
	if (_config != nullptr) {
		delete _config;
		_config = nullptr;
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

/*****************************************************************************
 * PVCore::PVConfig::get_lists_dir
 *****************************************************************************/

QString PVCore::PVConfig::get_lists_dir() const
{
	return _config_dir + QDir::separator() + _lists_folder;
}

/*****************************************************************************
 * PVCore::PVConfig::config
 *****************************************************************************/

QSettings& PVCore::PVConfig::config() const
{
	return *_config;
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
