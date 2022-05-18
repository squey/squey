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

#ifndef PVSQLTYPEMAP_FILE_H
#define PVSQLTYPEMAP_FILE_H

#include <memory>

#include <QString>
#include <QHash>

namespace PVRush
{

typedef QHash<int, QString> map_type;

class PVSQLTypeMap
{
  public:
	typedef std::shared_ptr<PVSQLTypeMap> p_type;

  public:
	static p_type get_map(QString const& driver);

  public:
	virtual QString map(int type) const = 0;
	virtual QString map_inendi(int type) const = 0;
};

typedef PVSQLTypeMap::p_type PVSQLTypeMap_p;

class PVSQLTypeMapMysql : public PVSQLTypeMap
{
  public:
	QString map(int type) const;
	QString map_inendi(int type) const;
};

class PVSQLTypeMapPostgres : public PVSQLTypeMap
{
  public:
	QString map(int type) const;
	QString map_inendi(int type) const;
};

class PVSQLTypeMapODBC : public PVSQLTypeMap
{
  public:
	QString map(int /*type*/) const { return "unknown"; }
	QString map_inendi(int /*type*/) const { return "string"; }
};

class PVSQLTypeMapSQLite : public PVSQLTypeMap
{
  public:
	QString map(int /*type*/) const { return "unknown"; }
	QString map_inendi(int /*type*/) const { return "string"; }
};
}

#endif
