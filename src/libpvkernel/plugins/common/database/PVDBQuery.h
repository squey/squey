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

#ifndef PVDBQUERY_FILE_H
#define PVDBQUERY_FILE_H

#include <pvbase/types.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <qlist.h>
#include <QString>
#include <QSqlQuery>
#include <QMetaType>
#include <QDomDocument>
#include <memory>
#include <string>
#include <vector>

#include "PVDBServ_types.h"

class QSettings;

namespace PVCore {
class PVSerializeObject;
}  // namespace PVCore

namespace PVRush
{

class PVDBQuery : public PVInputDescription
{
	friend class PVCore::PVSerializeObject;

  public:
	PVDBQuery();
	PVDBQuery(PVDBServ_p db);
	PVDBQuery(PVDBServ_p db, QString const& query);
	~PVDBQuery();

  public:
	virtual bool operator==(const PVInputDescription& other) const;

	void set_query(QString const& query) { _query = query; }
	QString const& get_query() const { return _query; }

	QString human_name() const;

	PVDBServ_p get_serv() { return _infos; };
	const PVDBServ_p get_serv() const { return _infos; };

	QSqlQuery to_query(chunk_index start, chunk_index nelts) const;

	QDomDocument get_format_from_db_schema() const;
	QList<QString> get_db_types() const;

	bool connect_serv();
	QString last_error_serv();

  public:
	virtual void save_to_qsettings(QSettings& settings) const;
	static std::unique_ptr<PVRush::PVInputDescription>
	load_from_string(std::vector<std::string> const&, bool multi_inputs);
	static std::vector<std::string> desc_from_qsetting(QSettings const& s);

  public:
	static std::unique_ptr<PVInputDescription> serialize_read(PVCore::PVSerializeObject& so);
	void serialize_write(PVCore::PVSerializeObject& so) const;

  protected:
	PVDBServ_p _infos;
	QString _query;
};
}

#endif
