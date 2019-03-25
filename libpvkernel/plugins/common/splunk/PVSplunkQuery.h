/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef __PVSPLUNKQUERY_FILE_H__
#define __PVSPLUNKQUERY_FILE_H__

#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/rush/PVInputDescription.h>

#include <QString>
#include <QMetaType>

#include <set>
#include "PVSplunkInfos.h"

namespace PVRush
{

class PVSplunkQuery : public PVInputDescription
{
  public:
	PVSplunkQuery(PVSplunkInfos const& infos, QString const& query, QString const& query_type);

  public:
	virtual bool operator==(const PVInputDescription& other) const;

	void set_query(QString const& query) { _query = query; }
	QString const& get_query() const { return _query; }

	void set_query_type(QString const& query_type) { _query_type = query_type; }
	QString const& get_query_type() const { return _query_type; }

	PVSplunkInfos& get_infos() { return _infos; }
	PVSplunkInfos const& get_infos() const { return _infos; }

	QString human_name() const;

  public:
	virtual void save_to_qsettings(QSettings& settings) const;
	static std::unique_ptr<PVRush::PVInputDescription>
	load_from_string(std::vector<std::string> const&, bool multi_inputs);
	static std::vector<std::string> desc_from_qsetting(QSettings const& s);

  public:
	static std::unique_ptr<PVRush::PVInputDescription>
	serialize_read(PVCore::PVSerializeObject& so);
	void serialize_write(PVCore::PVSerializeObject& so) const;

  protected:
	PVSplunkInfos _infos;
	QString _query;
	QString _query_type;
};
}

#endif // __PVSPLUNKQUERY_FILE_H__
