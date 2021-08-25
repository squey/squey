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

#ifndef PVOPCUAINFOS_H
#define PVOPCUAINFOS_H

#include <pvkernel/core/PVSerializeArchive.h>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/classification.hpp>

#include <QString>

#include <pvlogger.h>

namespace PVRush
{

class PVOpcUaInfos
{
  public:
	PVOpcUaInfos() : _port(0) {}
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

	inline bool operator==(PVOpcUaInfos const& o) const
	{
		return _host == o._host && _port == o._port && _index == o._index &&
		       _importer == o._importer && _format == o._format &&
		       _is_format_custom == o._is_format_custom && _filter_path == o._filter_path;
	}

	void serialize_write(PVCore::PVSerializeObject& so) const;
	static PVOpcUaInfos serialize_read(PVCore::PVSerializeObject& so);

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
} // namespace PVRush

#endif
