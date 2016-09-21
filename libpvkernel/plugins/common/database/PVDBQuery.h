/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVDBQUERY_FILE_H
#define PVDBQUERY_FILE_H

#include "PVDBServ_types.h"

#include <pvbase/types.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/rush/PVInputDescription.h>

#include <QString>
#include <QSqlQuery>
#include <QMetaType>

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

	QSqlQuery to_query(chunk_index start, chunk_index nelts) const;

	bool connect_serv();
	QString last_error_serv();

  public:
	virtual void save_to_qsettings(QSettings& settings) const;
	static std::unique_ptr<PVRush::PVInputDescription> load_from_string(std::string const&);
	static std::string desc_from_qsetting(QSettings const& s);

  public:
	static std::unique_ptr<PVInputDescription> serialize_read(PVCore::PVSerializeObject& so);
	void serialize_write(PVCore::PVSerializeObject& so) const;

  protected:
	PVDBServ_p _infos;
	QString _query;
};
}

#endif
