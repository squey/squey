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

#include "PVElasticsearchInfos.h"

#include <qcontainerfwd.h>
#include <qlist.h>
#include <memory>

#include "pvkernel/core/PVSerializeArchiveFixError.h"
#include "pvkernel/core/PVSerializeObject.h"

void PVRush::PVElasticsearchInfos::serialize_write(PVCore::PVSerializeObject& so) const
{
	so.attribute_write("host", _host);
	so.attribute_write("port", _port);
	so.attribute_write("index", _index);
	so.attribute_write("importer", _importer);
	so.attribute_write("format", _format);
	so.attribute_write("is_format_custom", _is_format_custom);
	so.attribute_write("filter_path", _filter_path);
	// Do not save credential as it is not encrypted
}

PVRush::PVElasticsearchInfos
PVRush::PVElasticsearchInfos::serialize_read(PVCore::PVSerializeObject& so)
{
	PVElasticsearchInfos info;
	info._host = so.attribute_read<QString>("host");
	info._port = so.attribute_read<uint16_t>("port");
	info._index = so.attribute_read<QString>("index");
	info._importer = so.attribute_read<QString>("importer");
	info._format = so.attribute_read<QString>("format");
	info._is_format_custom = so.attribute_read<bool>("is_format_custom");
	info._filter_path = so.attribute_read<QString>("filter_path");

	if (so.is_repaired_error()) {
		QStringList v = QString::fromStdString(so.get_repaired_value()).split(";");
		info._password = v[1];
		info._login = v[0];
	} else {
		throw PVCore::PVSerializeReparaibleCredentialError(
		    "Can't find credential for Elasticsearch", so.get_logical_path().toStdString());
	}

	return info;
}
