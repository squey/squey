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

#ifndef PVELASTICSEARCHQUERY_FILE_H
#define PVELASTICSEARCHQUERY_FILE_H

#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/rush/PVInputDescription.h>

#include <QString>
#include <QMetaType>

#include <set>
#include "PVElasticsearchInfos.h"

namespace PVRush
{

class PVElasticsearchQuery : public PVInputDescription
{
	friend class PVCore::PVSerializeObject;

  public:
	PVElasticsearchQuery(PVElasticsearchInfos const& infos,
	                     QString const& query,
	                     QString const& query_type);

  public:
	virtual bool operator==(const PVInputDescription& other) const;

	void set_query(QString const& query) { _query = query; }
	QString const& get_query() const { return _query; }

	void set_query_type(QString const& query_type) { _query_type = query_type; }
	QString const& get_query_type() const { return _query_type; }

	PVElasticsearchInfos& get_infos() { return _infos; }
	PVElasticsearchInfos const& get_infos() const { return _infos; }

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
	PVElasticsearchInfos _infos;
	QString _query;
	QString _query_type;
};
}

#endif
