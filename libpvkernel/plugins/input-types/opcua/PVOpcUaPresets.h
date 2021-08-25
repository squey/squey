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

#ifndef PVOPCUAPRESETS_H
#define PVOPCUAPRESETS_H

#include "../../common/opcua/PVOpcUaInfos.h"

#include <QString>
#include <QSettings>

#include <map>

namespace PVRush
{

class PVOpcUaPresets
{
  public:
	static const char PV_SETTINGS_INPUT[];

  private:
	PVOpcUaPresets();
	~PVOpcUaPresets();

  public:
	typedef unsigned int id_t;
	typedef std::map<id_t, QString> list_id_names_t;

  public:
	static PVOpcUaPresets& get();

  public:
	id_t add(QString const& name,
	         PVOpcUaInfos const& infos,
	         QString const& query,
	         QString const& query_type);
	void set(id_t id,
	         PVOpcUaInfos const& infos,
	         QString const& query,
	         QString const& query_type,
	         QString const& name = QString());
	list_id_names_t list_id_names();
	bool get(id_t id, PVOpcUaInfos& infos, QString& query, QString& query_type);
	void rm(id_t id);

  protected:
	QSettings _settings;
};
} // namespace PVRush

#endif
