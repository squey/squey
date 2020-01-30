/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVCONFIG_H
#define PVCORE_PVCONFIG_H

#include <QString>
#include <QVariant>

#include <memory>

class QSettings;

namespace PVCore
{

class PVConfig
{
  public:
	static constexpr const char* CONFIG_FILENAME = "config.ini";
	static constexpr const char* PRESETS_FILENAME = "presets.ini";

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

	QSettings& config() const; // TODO : remove me

	/**
	 * return a value from the local then global config
	 */
	static QVariant value(const QString& name);

	/**
	 * set a value to the local config
	 */
	static void set_value(const QString& name, const QVariant& value);

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

	/**
	 * return the name of the product (ex: pcap-inspector, inendi-inspector, etc)
	 */
	static std::string product_name();

	/**
	 * set the name of the product
	 */
	static void set_product_name(const std::string& product_name);

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
	QSettings* _local_config;
	QSettings* _global_config = nullptr;
	QString _username;
	std::string _product_name;
};
} // namespace PVCore

#endif /* PVCORE_PVCONFIG_H */
