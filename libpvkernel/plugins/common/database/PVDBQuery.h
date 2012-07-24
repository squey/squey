/**
 * \file PVDBQuery.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVDBQUERY_FILE_H
#define PVDBQUERY_FILE_H

#include "PVDBServ_types.h"

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/rush/PVInputDescription.h>

#include <QString>
#include <QSqlQuery>
#include <QMetaType>

#include <boost/shared_ptr.hpp>

namespace PVRush {

class PVDBQuery: public PVInputDescription
{
	friend class PVCore::PVSerializeObject;
public:
	PVDBQuery();
	PVDBQuery(PVDBServ_p db);
	PVDBQuery(PVDBServ_p db, QString const& query);
	~PVDBQuery();

	void set_query(QString const& query) { _query = query; }
	QString const& get_query() const { return _query; }

	QString human_name() const;

	PVDBServ_p get_serv() { return _infos; };

	QSqlQuery to_query(chunk_index start, chunk_index nelts) const;

	bool connect_serv();
	QString last_error_serv();
	
protected:
	void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);
	void serialize_write(PVCore::PVSerializeObject& so);

	PVSERIALIZEOBJECT_SPLIT

protected:
	PVDBServ_p _infos;
	QString _query;
};

}

#endif
