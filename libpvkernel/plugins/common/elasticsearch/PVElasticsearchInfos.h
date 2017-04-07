/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVELASTICSEARCHINFOS_H
#define PVELASTICSEARCHINFOS_H

#include <pvkernel/core/PVSerializeArchive.h>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/classification.hpp>

#include <QString>

#include <pvlogger.h>

namespace PVRush
{

class PVElasticsearchInfos
{
  public:
	PVElasticsearchInfos() : _port(0) {}
	void set_host(QString const& host) { _host = host; }
	void set_port(uint16_t port) { _port = port; }
	void set_index(QString const& index) { _index = index; }
	void set_importer(QString const& importer) { _importer = importer; }
	void set_login(QString const& login) { _login = login; }
	void set_password(QString const& password) { _password = password; }
	void set_format(QString const& format) { _format = format; }
	void set_custom_format(bool is_custom) { _is_format_custom = is_custom; }
	void set_filter_path(QString const& filter_path) { _filter_path = filter_path; }

	QString const& get_host() const { return _host; }
	uint16_t get_port() const { return _port; }
	QString const& get_index() const { return _index; }
	QString const& get_importer() const { return _importer; }
	QString const& get_login() const { return _login; }
	QString const& get_password() const { return _password; }
	QString const& get_format() const { return _format; }
	bool is_format_custom() const { return _is_format_custom; }
	QString const& get_filter_path() const { return _filter_path; }

	inline bool operator==(PVElasticsearchInfos const& o) const
	{
		return _host == o._host && _port == o._port && _index == o._index &&
		       _importer == o._importer && _format == o._format &&
		       _is_format_custom == o._is_format_custom && _filter_path == o._filter_path;
	}

	void serialize_write(PVCore::PVSerializeObject& so) const;
	static PVElasticsearchInfos serialize_read(PVCore::PVSerializeObject& so);

  private:
	QString _host;
	uint16_t _port;
	QString _index;
	QString _importer;
	QString _login;
	QString _password;
	QString _format;
	bool _is_format_custom = true;
	QString _filter_path;
};
}

#endif
