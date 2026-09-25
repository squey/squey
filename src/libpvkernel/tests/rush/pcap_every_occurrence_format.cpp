//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

// A field a pcap profile asks for at every occurrence is text.
//
// tshark joins every value a frame holds for such a field into one cell, separated
// by the aggregator: a frame carrying two ports writes "80|443", which a number
// column reads as nothing. Asked for at its first occurrence only, the field keeps
// the type it has.

#include <pvkernel/core/squey_assert.h>

// Before profileformat.h, which names rapidjson without including it.
#include <rapidjson/document.h>

#include "../../plugins/common/pcap/libpvpcap/include/libpvpcap/profileformat.h"

#include <QDomDocument>
#include <QDomElement>
#include <QDomNodeList>

#include <iostream>
#include <string>

#include "common.h"

namespace
{

//! A profile asking for the TCP port, at the given occurrence: "f" first, "a" all.
std::string profile(const std::string& occurrence)
{
	return R"json({
  "options": {
    "source": false, "destination": false, "protocol": false, "info": false,
    "filters": "", "header": false, "aggregator": "|", "occurrence": ")json" +
	       occurrence + R"json("
  },
  "children": [
    {
      "fields": [
        {"name": "Port", "filter_name": "tcp.port", "type": "FT_UINT16", "description": "", "select": true}
      ],
      "children": []
    }
  ]
})json";
}

//! The type of the axis the format gives the TCP port, or nothing if it has none.
QString port_type(const std::string& occurrence)
{
	rapidjson::Document json;
	json.Parse(profile(occurrence).c_str());
	PV_VALID(json.HasParseError(), false);

	const QDomDocument format = pvpcap::get_format(json, 1);
	const QDomNodeList axes = format.elementsByTagName("axis");
	for (int i = 0; i < axes.size(); ++i) {
		const QDomElement axis = axes.at(i).toElement();
		if (axis.attribute("name") == "tcp.port") {
			return axis.attribute("type");
		}
	}
	return {};
}

} // namespace

int main()
{
	pvtest::init_ctxt();

	const QString first = port_type("f");
	std::cout << "first occurrence: " << first.toStdString() << std::endl;
	PV_ASSERT_VALID(not first.isEmpty() and first != "string", "the port asked for once",
	                first.toStdString());

	const QString every = port_type("a");
	std::cout << "every occurrence: " << every.toStdString() << std::endl;
	PV_ASSERT_VALID(every == "string", "the port asked for at every occurrence",
	                every.toStdString());

	return 0;
}
