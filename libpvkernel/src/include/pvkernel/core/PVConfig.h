/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVCONFIG_H
#define PVCORE_PVCONFIG_H

#include <QString>

#include <memory>

class QSettings;

namespace PVCore
{

class PVConfig
{
	typedef std::shared_ptr<PVConfig> PVConfig_p;

  public:
	/**
	 * return a singleton on the global configuration accessor
	 */
	static PVConfig& get();

  public:
	/**
	 * return the path to the white/grey/black lists
	 */
	QString get_lists_dir() const;

	/**
	 * return a reference on the application's settings
	 */
	QSettings& config() const;

	/**
	 * return the current user name
	 */
	static QString username();

	/**
	 * return the path to the user configuration file
	 */
	static QString user_path();

	/**
	 * return the path to the user configuration directory
	 */
	static std::string user_dir();

  private:
	// private stuffs for a singleton declaration
	PVConfig();
	PVConfig(const PVConfig&);
	PVConfig& operator=(const PVConfig&);

  public:
	// public destructor to permit its destruction
	~PVConfig();

  private:
	static PVConfig_p _pvconfig;
	QSettings* _config;
	QString _username;
};
} // namespace PVCore

#endif /* PVCORE_PVCONFIG_H */
