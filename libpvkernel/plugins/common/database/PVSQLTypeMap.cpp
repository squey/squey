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
	return p_type();
}

QString PVRush::PVSQLTypeMapMysql::map(int type) const
{
	switch (type) {
	case FIELD_TYPE_TINY:
		return "tiny";
	case FIELD_TYPE_SHORT:
		return "short";
	case FIELD_TYPE_LONG:
		return "long";
	case FIELD_TYPE_INT24:
		return "int24";
	case FIELD_TYPE_YEAR:
		return "year";
	case FIELD_TYPE_LONGLONG:
		return "longlong";
	case FIELD_TYPE_FLOAT:
		return "float";
	case FIELD_TYPE_DOUBLE:
		return "double";
	case FIELD_TYPE_DECIMAL:
		return "decimal";
#if defined(FIELD_TYPE_NEWDECIMAL)
	case FIELD_TYPE_NEWDECIMAL:
		return "hexadecimal";
#endif
	case FIELD_TYPE_DATE:
		return "date";
	case FIELD_TYPE_TIME:
		return "time";
	case FIELD_TYPE_DATETIME:
		return "datetime";
	case FIELD_TYPE_TIMESTAMP:
		return "timestamp";
	case FIELD_TYPE_STRING:
		return "string";
	case FIELD_TYPE_VAR_STRING:
		return "varstring";
	case FIELD_TYPE_BLOB:
		return "blob";
	case FIELD_TYPE_TINY_BLOB:
		return "tinyblob";
	case FIELD_TYPE_MEDIUM_BLOB:
		return "mediumblob";
	case FIELD_TYPE_LONG_BLOB:
		return "longblob";
	case FIELD_TYPE_ENUM:
		return "enum";
	case FIELD_TYPE_SET:
		return "set";
	};
	return "unknown";
}

QString PVRush::PVSQLTypeMapMysql::map_inendi(int type) const
{
	switch (type) {
	case FIELD_TYPE_TINY:
	case FIELD_TYPE_SHORT:
	case FIELD_TYPE_LONG:
	case FIELD_TYPE_INT24:
	case FIELD_TYPE_YEAR:
	case FIELD_TYPE_LONGLONG:
		return "number_int32";

	case FIELD_TYPE_FLOAT:
	case FIELD_TYPE_DECIMAL:
		return "number_float";

	case FIELD_TYPE_DOUBLE:
		return "number_double";

	// TODO : dates must be propely supported
	case FIELD_TYPE_DATE:
	case FIELD_TYPE_TIME:
	case FIELD_TYPE_DATETIME:
	case FIELD_TYPE_TIMESTAMP:
		return "string";

	case FIELD_TYPE_STRING:
	case FIELD_TYPE_VAR_STRING:
	case FIELD_TYPE_BLOB:
	case FIELD_TYPE_TINY_BLOB:
	case FIELD_TYPE_MEDIUM_BLOB:
	case FIELD_TYPE_LONG_BLOB:
	case FIELD_TYPE_ENUM:
	case FIELD_TYPE_SET:
		return "string";
	};

	return "string";
}

QString PVRush::PVSQLTypeMapPostgres::map(int type) const
{
	switch (type) {
	case QBOOLOID:
		return "bool";
    case QINT2OID:
    case QINT4OID:
	case QINT8OID:
    case QOIDOID:
    case QREGPROCOID:
    case QXIDOID:
    case QCIDOID:
		return "integer";
	case QNUMERICOID:
    case QFLOAT4OID:
    case QFLOAT8OID:
		return "double";
	case QABSTIMEOID:
    case QRELTIMEOID:
    case QDATEOID:
		return "date";
	case QTIMEOID:
    case QTIMETZOID:
		return "time";
	case QTIMESTAMPOID:
    case QTIMESTAMPTZOID:
		return "datetime";
	case QBYTEAOID:
		return "bytearray";
	}
	return "unknown";
}

QString PVRush::PVSQLTypeMapPostgres::map_inendi(int type) const
{
	switch (type) {
	case QBOOLOID:
		return "string";
    case QINT2OID:
		return "number_int16";
	case QINT4OID:
	case QOIDOID:
	case QREGPROCOID:
	case QXIDOID:
	case QCIDOID:
		return "number_int32";
	case QINT8OID:
		return "number_int64";
	case QNUMERICOID:
    case QFLOAT4OID:
		return "number_float";
    case QFLOAT8OID:
		return "number_double";

	// TODO : dates must be propely supported
	case QABSTIMEOID:
    case QRELTIMEOID:
    case QDATEOID:
		return "string";
	case QTIMEOID:
    case QTIMETZOID:
		return "string";
	case QTIMESTAMPOID:
    case QTIMESTAMPTZOID:
		return "string";
	case QBYTEAOID:
		return "string";
	}
	return "string";
}
