/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVFormatVersion.h>
#include <pvkernel/rush/PVFormat_types.h>

#include <QDomNode>
#include <QStringList>

QString PVRush::PVFormatVersion::__impl::get_version(QDomDocument const& doc)
{
	return doc.documentElement().attribute("version", "0");
}

void PVRush::PVFormatVersion::to_current(QDomDocument& doc)
{
	QString version = __impl::get_version(doc);
	if (version == "0") {
		__impl::from0to1(doc);
		version = "1";
	}
	if (version == "1") {
		__impl::from1to2(doc);
		version = "2";
	}
	if (version == "2") {
		__impl::from2to3(doc);
		version = "3";
	}
	if (version == "3") {
		__impl::from3to4(doc);
		version = "4";
	}
	if (version == "4") {
		__impl::from4to5(doc);
		version = "5";
	}
	if (version == "5") {
		__impl::from5to6(doc);
		version = "6";
	}
}

void PVRush::PVFormatVersion::__impl::from0to1(QDomDocument& doc)
{
	_rec_0to1(doc.documentElement());
	doc.documentElement().setAttribute("version", "1");
}

void PVRush::PVFormatVersion::__impl::from1to2(QDomDocument& doc)
{
	_rec_1to2(doc.documentElement());
	doc.documentElement().setAttribute("version", "2");
}

void PVRush::PVFormatVersion::__impl::from2to3(QDomDocument& doc)
{
	_rec_2to3(doc.documentElement());
	doc.documentElement().setAttribute("version", "3");
}

void PVRush::PVFormatVersion::__impl::from3to4(QDomDocument& doc)
{
	_rec_3to4(doc.documentElement());
	doc.documentElement().setAttribute("version", "4");
}

void PVRush::PVFormatVersion::__impl::from4to5(QDomDocument& doc)
{
	_rec_4to5(doc.documentElement());
	doc.documentElement().setAttribute("version", "5");
}

void PVRush::PVFormatVersion::__impl::from5to6(QDomDocument& doc)
{
	_rec_5to6(doc.documentElement());
	doc.documentElement().setAttribute("version", "6");
}

void PVRush::PVFormatVersion::__impl::_rec_0to1(QDomElement elt)
{
	QString const& tag_name = elt.tagName();
	if (tag_name == "RegEx") {
		elt.setTagName("splitter");
		elt.setAttribute("type", "regexp");
		elt.setAttribute("regexp", elt.attribute("expression"));
		elt.removeAttribute("expression");
	} else if (tag_name == "url") {
		elt.setTagName("splitter");
		elt.setAttribute("type", "url");
	} else if (tag_name == "csv") {
		elt.setTagName("splitter");
		elt.setAttribute("type", "csv");
		elt.setAttribute("sep", elt.attribute("delimiter"));
		elt.removeAttribute("delimiter");
	} else if (tag_name == "filter") {
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
		_rec_0to1(children.at(i).toElement());
	}
}

void PVRush::PVFormatVersion::__impl::_rec_1to2(QDomElement elt)
{
	QString const& tag_name = elt.tagName();
	static QStringList tags = QStringList() << "protocol"
	                                        << "domain"
	                                        << "tld"
	                                        << "port"
	                                        << "url"
	                                        << "url-variables";
	static QStringList plottings = QStringList() << "default"
	                                             << "default"
	                                             << "default"
	                                             << "port"
	                                             << "minmax"
	                                             << "minmax";
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
		} else if (type == "regexp") {
			// Default dehavioru was to match the regular expression somewhere in the line
			elt.setAttribute("full-line", "false");
		}
	} else if (tag_name == "axis") {
		bool is_key = elt.attribute("key", "false") == "true";
		if (is_key) {
			QString cur_tag = elt.attribute("tag");
			if (!cur_tag.isEmpty()) {
				cur_tag += QString(QChar(':')) + "key";
			} else {
				cur_tag = "key";
			}
			elt.setAttribute("tag", cur_tag);
		}
		elt.removeAttribute("key");
	}

	QDomNodeList children = elt.childNodes();
	for (int i = 0; i < children.size(); i++) {
		_rec_1to2(children.at(i).toElement());
	}
}

void PVRush::PVFormatVersion::__impl::_rec_2to3(QDomElement elt)
{
	QString const& tag_name = elt.tagName();
	if (tag_name == "axis") {
		QString plotting = elt.attribute("plotting", "");
		QString type = elt.attribute("type", "");
		if (type != "time" && type != "ipv4" && plotting == "minmax") {
			// minmax is now default. if default was not minmax, it was only relevant for the time
			// and ipv4
			elt.setAttribute("plotting", "default");
		}
	}

	QDomNodeList children = elt.childNodes();
	for (int i = 0; i < children.size(); i++) {
		_rec_2to3(children.at(i).toElement());
	}
}

void PVRush::PVFormatVersion::__impl::_rec_3to4(QDomNode node)
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
				elt_mapping.setAttribute("time-format", QLatin1String("@PVTimeFormat(") +
				                                            elt.attribute("time-format", "") +
				                                            QLatin1String(")"));
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
}

void PVRush::PVFormatVersion::__impl::_rec_4to5(QDomNode node)
{
	if (node.isElement()) {
		QDomElement elt = node.toElement();
		if (elt.tagName() == "mapping") {
			QString tf = elt.attribute("time-format", QString());
			if (tf.size() > 0 && tf.startsWith("@PVTimeFormat(")) {
				tf = tf.mid(14, tf.size() - 15);
				elt.setAttribute("time-format", tf);
			}

			QString cl = elt.attribute("convert-lowercase", QString());
			if (cl.size() > 0 && cl.startsWith("@Bool(")) {
				cl = cl.mid(6, cl.size() - 7);
				elt.setAttribute("convert-lowercase", cl);
			}
		}
		if (elt.tagName() == "splitter") {
			QString sep = elt.attribute("sep", QString());
			if (sep.size() > 0 && sep.startsWith("@Char(")) {
				sep = sep.mid(6, sep.size() - 7);
				elt.setAttribute("sep", sep);
			}
		}
	}

	QDomNode child = node.firstChild();
	while (!child.isNull()) {
		_rec_4to5(child);
		child = child.nextSibling();
	}
}

void PVRush::PVFormatVersion::__impl::_rec_5to6(QDomNode node)
{
	if (node.isElement()) {
		QDomElement elt = node.toElement();
		if (elt.tagName() == "mapping") {
			QString tf = elt.attribute("time-format", QString());
			if (tf.size() > 0) {
				QDomElement axis_node = node.parentNode().toElement();
				axis_node.setAttribute("type_format", tf);
				elt.removeAttribute("time-format");
			}
		} else if (elt.tagName() == "axis") {
			// Move mapping from attribute to node
			QString mapping = elt.attribute("mapping", QString());
			QDomNode mapping_node;
			mapping_node.setNodeValue(mapping);
			elt.appendChild(mapping_node);
			elt.removeAttribute("mapping");

			// Move plotting from attribute to node
			QString plotting = elt.attribute("plotting", QString());
			QDomNode plotting_node;
			plotting_node.setNodeValue(plotting);
			elt.appendChild(plotting_node);
			elt.removeAttribute("plotting");

			// Move axis/time-format to axis/mapping/type_format
			QString tf = elt.attribute("time-format", QString());
			if (tf.size() > 0) {
				QDomElement axis_node = node.toElement();
				axis_node.setAttribute("type_format", tf);
				elt.removeAttribute("time-format");
			}
		}
	}

	QDomNode child = node.firstChild();
	while (!child.isNull()) {
		_rec_5to6(child);
		child = child.nextSibling();
	}
}
