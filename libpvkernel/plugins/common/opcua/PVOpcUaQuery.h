/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVOPCUAQUERY_FILE_H
#define PVOPCUAQUERY_FILE_H

#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include "PVOpcUaInfos.h"

#include <QString>
#include <QMetaType>

#include <set>

namespace PVRush
{

class PVOpcUaQuery : public PVInputDescription
{
	friend class PVCore::PVSerializeObject;

  public:
	PVOpcUaQuery(PVOpcUaInfos const& infos, QString const& query, QString const& query_type);

  public:
	virtual bool operator==(const PVInputDescription& other) const override;

	PVOpcUaInfos const& infos() const { return _infos; };

	void set_query(QString const& query) { _query = query; }
	QString const& get_query() const { return _query; }

	void set_query_type(QString const& query_type) { _query_type = query_type; }
	QString const& get_query_type() const { return _query_type; }

	QString human_name() const override;

  public:
	virtual void save_to_qsettings(QSettings& settings) const override;
	static std::unique_ptr<PVRush::PVInputDescription>
	load_from_string(std::vector<std::string> const&, bool multi_inputs);
	static std::vector<std::string> desc_from_qsetting(QSettings const& s);

  public:
	static std::unique_ptr<PVRush::PVInputDescription>
	serialize_read(PVCore::PVSerializeObject& so);
	void serialize_write(PVCore::PVSerializeObject& so) const override;

  protected:
	PVOpcUaInfos _infos;
	QString _query;
	QString _query_type;
};
} // namespace PVRush

#endif