/* * MIT License
 *
 * © ESI Group, 2015
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

#ifndef PVCORE_PVCONFIG_H
#define PVCORE_PVCONFIG_H

#include <QString>
#include <QVariant>
#include <memory>
#include <string>

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
	static QString user_dir();

	/**
	 * return the name of the product (ex: squey, etc)
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
