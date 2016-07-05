#include "common.h"

#include <pvkernel/rush/PVFormatVersion.h>
#include <pvkernel/rush/PVUtils.h>

#include <pvkernel/core/inendi_assert.h>

#include <QDomElement>
#include <QDomDocument>

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
