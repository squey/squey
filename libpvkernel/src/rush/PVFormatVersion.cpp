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
		if (!from2to3(doc)) {
			return false;
		}
		if (!from3to4(doc)) {
			return false;
		}
	}
	if (version == "1") {
		if (!from1to2(doc)) {
			return false;
		}
		if (!from2to3(doc)) {
			return false;
		}
		if (!from3to4(doc)) {
			return false;
		}
	}
	if (version == "2") {
		if (!from2to3(doc)) {
			return false;
		}
		if (!from3to4(doc)) {
			return false;
		}
	}
	if (version == "3") {
		if (!from3to4(doc)) {
			return false;
		}
	}
	if (version != "4") {
		return false;
	}

	doc.documentElement().setAttribute("version", PVFORMAT_CURRENT_VERSION);
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

bool PVRush::PVFormatVersion::from2to3(QDomDocument& doc)
{
	return _rec_2to3(doc.documentElement());
}

bool PVRush::PVFormatVersion::from3to4(QDomDocument& doc)
{
	return _rec_3to4(doc.documentElement());
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
	static QStringList plottings = QStringList() << "default" << "default" << "default" << "port" << "minmax" << "minmax";
	if (tag_name == "splitter") {
		QString type = elt.attribute("type", "");
		if (type == "url") {
			// Set default axes tags and plottings
			QDomNodeList children = elt.childNodes();
			for (unsigned int i = 0; i < 6; i++) {
				QDomElement c_elt = children.at(i).toElement();
				if (c_elt.tagName() == "field") {
					// Take the axis
					QDomElement axis = c_elt.firstChildElement("axis");
					// and set the default tag
					axis.setAttribute("tag", tags[i]);
					axis.setAttribute("plotting", plottings[i]);
				}
			}
		}
		else
		if (type == "regexp") {
			// Default dehavioru was to match the regular expression somewhere in the line
			elt.setAttribute("full-line", "false");
		}
	}
	else
	if (tag_name == "axis") {
		bool is_key = elt.attribute("key", "false") == "true";
		if (is_key) {
			QString cur_tag = elt.attribute("tag");
			if (!cur_tag.isEmpty()) {
				cur_tag += QString(QChar(':')) + "key";
			}
			else {
				cur_tag = "key";
			}
			elt.setAttribute("tag", cur_tag);
		}
		elt.removeAttribute("key");
	}

	QDomNodeList children = elt.childNodes();
	for (int i = 0; i < children.size(); i++) {
		if (!_rec_1to2(children.at(i).toElement())) {
			return false;
		}
	}
	return true;
}

bool PVRush::PVFormatVersion::_rec_2to3(QDomElement elt)
{
	QString const& tag_name = elt.tagName();
	if (tag_name == "axis") {
		QString plotting = elt.attribute("plotting", "");
		QString type = elt.attribute("type", "");
		if (type != "time" && type != "ipv4" && plotting == "minmax") {
			// minmax is now default. if default was not minmax, it was only relevant for the time and ipv4
			elt.setAttribute("plotting", "default");
		}
	}

	QDomNodeList children = elt.childNodes();
	for (int i = 0; i < children.size(); i++) {
		if (!_rec_2to3(children.at(i).toElement())) {
			return false;
		}
	}
	return true;
}

bool PVRush::PVFormatVersion::_rec_3to4(QDomNode node)
{
	if (node.isElement()) {
		QDomElement elt = node.toElement();
		if (elt.tagName() == "axis") {
			QString mapping = elt.attribute("mapping", "");
			QString plotting = elt.attribute("plotting", "");
			QString type = elt.attribute("type", "");
			QDomElement elt_mapping = elt.ownerDocument().createElement(PVFORMAT_XML_TAG_MAPPING);
			elt_mapping.setAttribute(PVFORMAT_MAP_PLOT_MODE_STR, mapping);
			if (type == "time") {
				elt_mapping.setAttribute("time-format", QLatin1String("@PVTimeFormat(") + elt.attribute("time-format", "") + QLatin1String(")"));
				elt.removeAttribute("time-format");
			}
			elt.appendChild(elt_mapping);

			QDomElement elt_plotting = elt.ownerDocument().createElement(PVFORMAT_XML_TAG_PLOTTING);
			elt_plotting.setAttribute(PVFORMAT_MAP_PLOT_MODE_STR, plotting);
			elt.appendChild(elt_plotting);

			elt.removeAttribute("mapping");
			elt.removeAttribute("plotting");
		}
	}

	QDomNode child = node.firstChild();
	while (!child.isNull()) {
		_rec_3to4(child);
		child = child.nextSibling();
	}

	return true;
}

