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

#ifndef __PVSPLUNKINFOS_H__
#define __PVSPLUNKINFOS_H__

#include <pvkernel/core/PVSerializeArchive.h>

#include <QString>

namespace PVRush
{

class PVSplunkInfos
{
  public:
	void set_host(QString const& host) { _host = host; }
	void set_port(uint16_t port) { _port = port; }
	void set_login(QString const& login) { _login = login; }
	void set_password(QString const& password) { _password = password; }

	void set_splunk_index(QString const& index) { _splunk_index = index; }
	void set_splunk_host(QString const& host) { _splunk_host = host; }
	void set_splunk_sourcetype(QString const& sourcetype) { _splunk_sourcetype = sourcetype; }
	void set_custom_format(bool is_custom) { _is_format_custom = is_custom; }
	void set_format(QString const& format) { _format = format; }

	QString const& get_host() const { return _host; }
	uint16_t get_port() const { return _port; }
	QString const& get_login() const { return _login; }
	QString const& get_password() const { return _password; }

	QString const& get_splunk_index() const { return _splunk_index; }
	QString const& get_splunk_host() const { return _splunk_host; }
	QString const& get_splunk_sourcetype() const { return _splunk_sourcetype; }
	bool is_format_custom() const { return _is_format_custom; }
	QString const& get_format() const { return _format; }

	inline bool operator==(PVSplunkInfos const& o) const
	{
		return _host == o._host && _port == o._port && _splunk_index == o._splunk_index &&
		       _splunk_host == o._splunk_host && _splunk_sourcetype == o._splunk_sourcetype &&
		       _format == o._format;
	}

	void serialize_write(PVCore::PVSerializeObject& so) const;
	static PVSplunkInfos serialize_read(PVCore::PVSerializeObject& so);

  private:
	QString _host;
	uint16_t _port;
	QString _login;
	QString _password;

	QString _splunk_index;
	QString _splunk_host;
	QString _splunk_sourcetype;
	bool _is_format_custom = true;
	QString _format;
};
}

#endif // __PVSPLUNKINFOS_H__
