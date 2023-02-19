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

#ifndef PVDBINFOS_FILE_H
#define PVDBINFOS_FILE_H

#include <pvkernel/core/PVSerializeArchive.h>

#include <QSqlDatabase>
#include <QString>

#include "PVDBServ_types.h"

namespace PVRush
{

class PVDBQuery;
class PVDBServ;

class PVDBInfos
{
	friend class PVDBServ;
	friend class PVCore::PVSerializeObject;

  public:
	PVDBInfos();
	PVDBInfos(QString const& type,
	          QString const& host,
	          uint16_t port,
	          QString const& username,
	          QString const& password,
	          QString const& dbname,
	          QString const& options = QString(""));

  public:
	void set_type(QString const& type) { _type = type; }
	void set_host(QString const& host) { _host = host; }
	void set_username(QString const& username) { _username = username; }
	void set_password(QString const& password) { _password = password; }
	void set_options(QString const& options) { _options = options; }
	void set_dbname(QString const& dbname) { _dbname = dbname; }
	void set_port(uint16_t port) { _port = port; }

	QString const& get_type() const { return _type; }
	QString const& get_host() const { return _host; }
	QString const& get_username() const { return _username; }
	QString const& get_password() const { return _password; }
	QString const& get_options() const { return _options; }
	QString const& get_dbname() const { return _dbname; }
	uint16_t get_port() const { return _port; }

	// That name *must* be unique accross different databases
	QString database_name() const;

	void serialize_write(PVCore::PVSerializeObject& so) const;
	static PVDBInfos serialize_read(PVCore::PVSerializeObject& so);

  protected:
	QString _host;
	QString _username;
	QString _password;
	QString _options;
	QString _type;
	QString _dbname;
	uint16_t _port = 0;
};

class PVDBServ : public PVDBInfos
{
	friend class PVDBQuery;

  public:
	typedef PVDBServ_p p_type;

  public:
	PVDBServ(PVDBInfos const& infos);
	~PVDBServ();

  public:
	bool connect();
	QString last_error() const;
	QSqlDatabase to_database();

  private:
	QSqlDatabase _db;
};
}

#endif
