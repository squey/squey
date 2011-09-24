#include <pvkernel/rush/PVFormatVersion.h>
#include <QDomNode>

QString PVRush::PVFormatVersion::get_version(QDomDocument const& doc)
{
	return doc.documentElement().attribute("version", "0");
}

bool PVRush::PVFormatVersion::to_current(QDomDocument& doc)
{
	QString version = get_version(doc);
	if (version == "0") {
		return from0to1(doc);
	}
	return version == "1";
}

bool PVRush::PVFormatVersion::from0to1(QDomDocument& doc)
{
	return _rec_0to1(doc.documentElement());
}

bool PVRush::PVFormatVersion::_rec_0to1(QDomElement elt)
{
	QString const& tag_name = elt.tagName();
	if (tag_name == "RegEx") {
		elt.setTagName("splitter");
		elt.setAttribute("type", "regexp");
		elt.setAttribute("regexp", elt.attribute("expression"));
		elt.removeAttribute("expression");
	}
	else
	if (tag_name == "url") {
		elt.setTagName("splitter");
		elt.setAttribute("type", "url");
	}
	else
	if (tag_name == "pcap") {
		elt.setTagName("splitter");
		elt.setAttribute("type", "pcap");
	}
	else
	if (tag_name == "csv") {
		elt.setTagName("splitter");
		elt.setAttribute("type", "csv");
		elt.setAttribute("sep", elt.attribute("delimiter"));
		elt.removeAttribute("delimiter");
	}
	else
	if (tag_name == "filter") {
		if (elt.attribute("type") == "include") {
			elt.setAttribute("reverse", "0");
		} else {
			elt.setAttribute("reverse", "1");
		}
		elt.setAttribute("type", "regexp");
		elt.setAttribute("regexp", elt.attribute("expression"));
		elt.removeAttribute("expression");
		elt.removeAttribute("type");

	}

	QDomNodeList children = elt.childNodes();
	for (int i = 0; i < children.size(); i++) {
		if (!_rec_0to1(children.at(i).toElement())) {
			return false;
		}
	}
	return true;
}
