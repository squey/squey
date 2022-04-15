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

#include "common.h"

#include <pvkernel/rush/PVFormatVersion.h>
#include <pvkernel/rush/PVUtils.h>

#include <pvkernel/core/inendi_assert.h>

#include <QDomDocument>
#include <QDomElement>
#include <QTextStream>
#include <QHashSeed>

static constexpr const char* xml_origin = TEST_FOLDER "/pvkernel/rush/formats/apache.access.format";
static constexpr const char* xml_ref =
    TEST_FOLDER "/pvkernel/rush/formats_ref/apache.access.format";

QDomDocument get_dom_from_filename(std::string const& filename)
{
	QDomDocument xml;
	QFile file(filename.c_str());
	if (!file.open(QIODevice::ReadOnly)) {
		throw std::runtime_error("Invalid file");
	}
	if (!xml.setContent(&file)) {
		file.close();
		throw std::runtime_error("Invalid file");
	}
	file.close();
	return xml;
}

int main()
{
	// We need to set this as we dump QDom which use a QHash. Without a seed, printing is random.
	QHashSeed::setDeterministicGlobalSeed();

	QDomDocument xml = get_dom_from_filename(xml_origin);

	PVRush::PVFormatVersion::to_current(xml);

	std::string res_file = pvtest::get_tmp_filename();
	QFile res(res_file.c_str());
	if (!res.open(QIODevice::WriteOnly | QIODevice::Text)) {
		throw std::runtime_error("Invalid file res");
	}
	QTextStream stream(&res);

	xml.documentElement().save(stream, QDomNode::CDATASectionNode /* = 4 */);

	res.close();

	std::cout << res_file << "/" << xml_ref << std::endl;
	PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(res_file, xml_ref));

	return 0;
}
