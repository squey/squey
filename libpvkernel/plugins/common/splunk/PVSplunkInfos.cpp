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

#include "PVSplunkInfos.h"

void PVRush::PVSplunkInfos::serialize_write(PVCore::PVSerializeObject& so) const
{
	so.attribute_write("host", _host);
	so.attribute_write("port", _port);
	so.attribute_write("splunk_index", _splunk_index);
	so.attribute_write("splunk_host", _splunk_host);
	so.attribute_write("splunk_sourcetype", _splunk_sourcetype);
	so.attribute_write("is_format_custom", _is_format_custom);
	so.attribute_write("format", _format);
}

PVRush::PVSplunkInfos PVRush::PVSplunkInfos::serialize_read(PVCore::PVSerializeObject& so)
{
	PVRush::PVSplunkInfos infos;
	infos._host = so.attribute_read<QString>("host");
	infos._port = so.attribute_read<uint16_t>("port");
	infos._splunk_index = so.attribute_read<QString>("splunk_index");
	infos._splunk_host = so.attribute_read<QString>("splunk_host");
	infos._splunk_sourcetype = so.attribute_read<QString>("splunk_sourcetype");
	infos._is_format_custom = so.attribute_read<bool>("is_format_custom");
	infos._format = so.attribute_read<QString>("format");
	return infos;
}
