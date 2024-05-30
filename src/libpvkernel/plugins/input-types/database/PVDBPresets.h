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

#ifndef PVDBPRESETS_H
#define PVDBPRESETS_H

#include "../../common/database/PVDBInfos.h"

#include <QString>
#include <QSettings>

#include <map>

#define PV_SETTINGS_INPUT_DB "plugins/input_type/database/"

namespace PVRush
{

class PVDBPresets
{
  private:
	PVDBPresets();
	~PVDBPresets();

  public:
	typedef unsigned int id_t;
	typedef std::map<id_t, QString> list_id_names_t;

  public:
	static PVDBPresets& get();

  public:
	id_t add(QString const& name, PVDBInfos const& infos, QString const& query);
	void
	set(id_t id, PVDBInfos const& infos, QString const& query, QString const& name = QString());
	list_id_names_t list_id_names();
	bool get(id_t id, PVDBInfos& infos, QString& query);
	void rm(id_t id);

  protected:
	QSettings _settings;
};
}

#endif
