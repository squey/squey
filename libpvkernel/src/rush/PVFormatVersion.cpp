#include <pvkernel/rush/PVFormatVersion.h>
#include <pvkernel/rush/PVFormat_types.h>

#include <QDomNode>
#include <QStringList>

QString PVRush::PVFormatVersion::get_version(QDomDocument const& doc)
{
	return doc.documentElement().attribute("version", "0");
}

bool PVRush::PVFormatVersion::to_current(QDomDocument& doc)
{
	QString version = get_version(doc);
	if (version == "0") {
		if (!from0to1(doc)) {
			return false;
		}
		if (!from1to2(doc)) {
			return false;
		}
	}
	if (version == "1") {
		if (!from1to2(doc)) {
			return false;
		}
	}

	if (version != "2") {
		return false;
	}

	doc.documentElement().setAttribute("version", "2");
	return true;
}

bool PVRush::PVFormatVersion::from0to1(QDomDocument& doc)
{
	return _rec_0to1(doc.documentElement());
}

bool PVRush::PVFormatVersion::from1to2(QDomDocument& doc)
{
	return _rec_1to2(doc.documentElement());
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

bool PVRush::PVFormatVersion::_rec_1to2(QDomElement elt)
{
	QString const& tag_name = elt.tagName();
	static QStringList tags = QStringList() << "protocol" << "domain" << "tld" << "port" << "url" << "url-variables";
	if (tag_name == "splitter") {
		QString type = elt.attribute("type", "");
		if (type == "url") {
			// Set default axes tags
			QDomNodeList children = elt.childNodes();
			for (unsigned int i = 0; i < 6; i++) {
				QDomElement c_elt = children.at(i).toElement();
				if (c_elt.tagName() == "field") {
					// Take the axis
					QDomElement axis = c_elt.firstChildElement("axis");
					// and set the default tag
					axis.setAttribute(PVFORMAT_AXIS_TAG_STR, tags[i]);
				}
			}
		}
		else
		if (type == "regexp") {
			// Default dehavioru was to match the regular expression somewhere in the line
			elt.setAttribute("full-line", "false");
		}
	}

	QDomNodeList children = elt.childNodes();
	for (int i = 0; i < children.size(); i++) {
		if (!_rec_1to2(children.at(i).toElement())) {
			return false;
		}
	}
	return true;
}
