//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include "PVSQLTypeMap.h"
#include "mysql_types.h"
#include "postgresql_types.h"

#include <QMetaType>

PVRush::PVSQLTypeMap_p PVRush::PVSQLTypeMap::get_map(QString const& driver)
{
	if (driver == "QMYSQL") {
		return p_type(new PVSQLTypeMapMysql());
	}
	if (driver == "QPSQL") {
		return p_type(new PVSQLTypeMapPostgres());
	}
	if (driver == "QSQLITE") {
		return p_type(new PVSQLTypeMapSQLite());
	}
	if (driver == "QODBC") {
		return p_type(new PVSQLTypeMapODBC());
	}
	return {};
}

QString PVRush::PVSQLTypeMapMysql::map(int type) const
{
	switch (type) {
	case QMetaType::Short:
		return "SMALLINT";
	case QMetaType::UShort:
		return "SMALLINT UNSIGNED";
	case QMetaType::Int:
		return "MEDIUMINT";
	case QMetaType::UInt:
		return "MEDIUMINT USIGNED";
	case QMetaType::Long:
		return "INT";
	case QMetaType::ULong:
		return "INT UNSIGNED";
	case QMetaType::LongLong:
		return "BIGINT";
	case QMetaType::ULongLong:
		return "BIGINT UNSIGNED";
	case QMetaType::Float:
		return "float";
	case QMetaType::Double:
		return "double";
	case QMetaType::QDate:
		return "DATE";
	case QMetaType::QDateTime:
		return "DATETIME";
	case QMetaType::QString:
		return "CHAR";
	};
	return "unknown";
}

QPair<QString, QString> PVRush::PVSQLTypeMapMysql::map_squey(int type) const
{
	switch (type) {
		case QMetaType::Short:
			return { "number_int16", "" };
		case QMetaType::UShort:
			return { "number_uint16", "" };
		case QMetaType::Int:
			return { "number_int32", "" };
		case QMetaType::UInt:
			return { "number_uint32", "" };
		case QMetaType::LongLong:
			return { "number_int64", "" };
		case QMetaType::ULongLong:
			return { "number_uint64", "" };
		case QMetaType::Float:
			return { "number_float", "" };
		case QMetaType::Double:
			return { "number_double", "" };
		case QMetaType::QDate:
			return { "time", "yyyy-M-d" };
		case QMetaType::QDateTime:
			return { "time", "yyyy-M-d HH:m:ss.S" };
		case QMetaType::QString:
			return { "string", "" };
		default:
			return { "string", "" };
	};
}

QString PVRush::PVSQLTypeMapPostgres::map(int type) const
{
	switch (type) {
	case QMetaType::Short:
		return "SMALLINT";
	case QMetaType::UShort:
		return "SMALLINT UNSIGNED";
	case QMetaType::Int:
		return "MEDIUMINT";
	case QMetaType::UInt:
		return "MEDIUMINT USIGNED";
	case QMetaType::Long:
		return "INT";
	case QMetaType::ULong:
		return "INT UNSIGNED";
	case QMetaType::LongLong:
		return "BIGINT";
	case QMetaType::ULongLong:
		return "BIGINT UNSIGNED";
	case QMetaType::Float:
		return "float";
	case QMetaType::Double:
		return "double";
	case QMetaType::QDate:
		return "DATE";
	case QMetaType::QDateTime:
		return "DATETIME";
	case QMetaType::QString:
		return "CHAR";
	};
	return "unknown";
}

QPair<QString, QString> PVRush::PVSQLTypeMapPostgres::map_squey(int type) const
{
	switch (type) {
		case QMetaType::Short:
			return { "number_int16", "" };
		case QMetaType::UShort:
			return { "number_uint16", "" };
		case QMetaType::Int:
			return { "number_int32", "" };
		case QMetaType::UInt:
			return { "number_uint32", "" };
		case QMetaType::LongLong:
			return { "number_int64", "" };
		case QMetaType::ULongLong:
			return { "number_uint64", "" };
		case QMetaType::Float:
			return { "number_float", "" };
		case QMetaType::Double:
			return { "number_double", "" };
		case QMetaType::QDate:
			return { "time", "yyyy-M-d" };
		case QMetaType::QDateTime:
			return { "time", "yyyy-M-d HH:m:ss.S" };
		case QMetaType::QString:
			return { "string", "" };
		default:
			return { "string", "" };
	}
}
