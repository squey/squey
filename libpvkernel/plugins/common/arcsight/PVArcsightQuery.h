/**
 * \file PVArcsightInfos.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVARCSIGHTQUERY_FILE_H
#define PVARCSIGHTQUERY_FILE_H

#include "PVArcsightInfos.h"

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/rush/PVInputDescription.h>

#include <QString>
#include <QMetaType>


#include <set>

namespace PVRush {

class PVArcsightQuery: public PVInputDescription
{
	friend class PVCore::PVSerializeObject;
public:
	typedef std::set<uint32_t> fields_indexes_t;
public:
	PVArcsightQuery();
	PVArcsightQuery(PVArcsightInfos const& infos);
	PVArcsightQuery(PVArcsightInfos const& infos, QString const& query);
	~PVArcsightQuery();

public:
	virtual bool operator==(const PVInputDescription& other) const;

	void set_start_ms(int64_t start_ms) { _start_ms = start_ms; }
	int64_t get_start_ms() const { return _start_ms; }

	void set_end_ms(int64_t end_ms) { _end_ms = end_ms; }
	int64_t get_end_ms() const { return _end_ms; }

	void set_query(QString const& query) { _query = query; }
	QString const& get_query() const { return _query; }

	PVArcsightInfos& get_infos() { return _infos; }
	PVArcsightInfos const& get_infos() const { return _infos; }

	QString human_name() const;

	bool keep_all_fields() const { return _fields_kept.size() == 0; }

	fields_indexes_t const& get_fields_kept() const { return _fields_kept; }
	fields_indexes_t& get_fields_kept() { return _fields_kept; }
	
public:
	virtual void save_to_qsettings(QSettings& settings) const;
	virtual void load_from_qsettings(const QSettings& settings);

protected:
	void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);
	void serialize_write(PVCore::PVSerializeObject& so);

	PVSERIALIZEOBJECT_SPLIT

protected:
	PVArcsightInfos _infos;
	QString _query;
	int64_t _start_ms;
	int64_t _end_ms;
	fields_indexes_t _fields_kept;
};

}

#endif
