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

#include <pvkernel/rush/PVFormatVersion.h>
#include <pvkernel/rush/PVFormat_types.h>
#include <qcontainerfwd.h>
#include <qlatin1stringview.h>
#include <qlist.h>
#include <cassert>
#include <QChar>
#include <QString>
#include <vector>

// Utility function to convert pre 7 format to 7 format.for mapping/type/plotting
static QString const get_type_from_format(QString const& type_attr, QString const& mapped_attr)
{
	if (type_attr == "integer" and mapped_attr == "unsigned")
		return "number_uint32";
	else if (type_attr == "integer" and mapped_attr == "hexadecimal")
		return "number_uint32";
	else if (type_attr == "integer" and mapped_attr == "octal")
		return "number_uint32";
	else if (type_attr == "integer" and mapped_attr == "default")
		return "number_int32";
	else if (type_attr == "host" and mapped_attr == "default")
		return "string";
	else if (type_attr == "enum" and mapped_attr == "default")
		return "string";
	else if (type_attr == "float")
		return "number_float";
	return type_attr;
}

static QString const get_mapped_from_format(QString const& type_attr, QString const& mapped_attr)
{
	if (type_attr == "integer" and mapped_attr == "unsigned")
		return "default";
	else if (type_attr == "integer" and mapped_attr == "hexadecimal")
		return "default";
	else if (type_attr == "integer" and mapped_attr == "octal")
		return "default";
	else if (type_attr == "integer" and mapped_attr == "default")
		return "default";
	else if (type_attr == "host" and mapped_attr == "default")
		return "host";
	else if (type_attr == "enum" and mapped_attr == "default")
		return "default";
	else if (type_attr == "string" and mapped_attr == "default")
		return "string";
	else if (type_attr == "ipv4" and mapped_attr == "uniform")
		return "default";
	return mapped_attr;
}

static QString const get_scaled_from_format(QString const& type_attr,
                                             QString const& mapped_attr,
                                             QString const& scaled_attr)
{
	if (type_attr == "enum")
		return "enum";
	else if (type_attr == "ipv4" and mapped_attr == "uniform")
		return "enum";
	else if (scaled_attr == "minmax")
		return "default";
	return scaled_attr;
}

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
	if (version == "6") {
		__impl::from6to7(doc);
		version = "7";
	}
	if (version == "7") {
		__impl::from7to8(doc);
		version = "8";
	}
	if (version == "8") {
		__impl::from8to9(doc);
		version = "9";
	}
	if (version == "9") {
		__impl::from9to10(doc);
		version = "10";
	}
	if (version == "10") {
		__impl::from10to11(doc);
		version = "11";
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

void PVRush::PVFormatVersion::__impl::from6to7(QDomDocument& doc)
{
	QDomNodeList axis = doc.documentElement().elementsByTagName("axis");
	for (int i = 0; i < axis.size(); i++) {
		QDomElement ax = axis.at(i).toElement();
		// Update type value
		QString type = ax.attribute("type");
		if (type.isNull()) {
			type = "string";
		}
		// Update mapping value
		QDomElement mapped = ax.namedItem("mapping").toElement();
		QString mapping = mapped.attribute("mode");
		if (mapping.isNull()) {
			mapping = "default";
		}
		// Update plotting value
		QDomElement scaled = ax.namedItem("plotting").toElement();
		QString plotting = scaled.attribute("mode");
		if (plotting.isNull()) {
			plotting = "default";
		}
		scaled.toElement().setAttribute("mode", get_scaled_from_format(type, mapping, plotting));
		mapped.toElement().setAttribute("mode", get_mapped_from_format(type, mapping));
		ax.setAttribute("type", get_type_from_format(type, mapping));
		// Update type_format
		if (mapping == "hexadecimal") {
			ax.toElement().setAttribute("type_format", "%#x");
		} else if (mapping == "octal") {
			ax.toElement().setAttribute("type_format", "%#o");
		}
		// Remove extra mapped node
		auto mappings = ax.elementsByTagName("mapping");
		for (int j = 1; j < mappings.count(); j++) {
			ax.removeChild(mappings.at(j));
		}
		// Remove extra scaled node
		auto plottings = ax.elementsByTagName("plotting");
		for (int j = 1; j < plottings.count(); j++) {
			ax.removeChild(plottings.at(j));
		}
		// Remove time-sample attribute
		ax.removeAttribute("time-sample");
	}
	doc.documentElement().setAttribute("version", "7");
}

void PVRush::PVFormatVersion::__impl::from7to8(QDomDocument& doc)
{
	QDomNodeList splitter = doc.documentElement().elementsByTagName("splitter");
	for (int i = 0; i < splitter.size(); i++) {
		QDomElement ax = splitter.at(i).toElement();
		// Update type value
		QString type = ax.attribute("type");
		if (type != "url") {
			continue;
		}

		std::vector<int> pos(10, -1);

		auto fields = ax.childNodes();
		for (int j = 0; j < fields.size(); j++) {
			QDomElement axis = fields.at(j).namedItem("axis").toElement();
			if (axis.isNull()) {
				// Field without axis is like "no field" in url splitter
				continue;
			}
			QString tag = axis.attribute("tag");
			if (tag.contains("protocol")) {
				pos[j] = 0;
			} else if (tag.contains("subdomain")) {
				pos[j] = 1;
			} else if (tag.contains("host")) {
				pos[j] = 2;
			} else if (tag.contains("domain")) {
				pos[j] = 3;
			} else if (tag.contains("tld")) {
				pos[j] = 4;
			} else if (tag.contains("port")) {
				pos[j] = 5;
			} else if (tag.contains("url-variables")) {
				pos[j] = 7;
			} else if (tag.contains("url-credentials")) {
				pos[j] = 9;
			} else if (tag.contains("url-anchor")) {
				pos[j] = 8;
			} else {
				assert(tag == "url");
				// CHeck it at the end to avoid mismatch with variables, cred...
				pos[j] = 6;
			}
		}

		QDomElement new_splitter = ax.ownerDocument().createElement("splitter");
		new_splitter.setAttribute("type", "url");

		for (int p : pos) {
			if (p == -1) {
				QDomElement new_field = ax.ownerDocument().createElement("field");
				new_splitter.appendChild(new_field);
			} else {
				new_splitter.appendChild(fields.at(p).cloneNode());
			}
		}

		splitter.at(i).parentNode().replaceChild(new_splitter, splitter.at(i));
	}

	QDomNodeList axis = doc.documentElement().elementsByTagName("axis");
	for (int i = 0; i < axis.size(); i++) {
		QDomElement ax = axis.at(i).toElement();
		ax.removeAttribute("tag");
		ax.removeAttribute("key");
		ax.removeAttribute("group");
	}

	doc.documentElement().setAttribute("version", "8");
}

void PVRush::PVFormatVersion::__impl::from8to9(QDomDocument& doc)
{
	QDomNodeList converters = doc.documentElement().elementsByTagName("converter");
	for (int i = 0; i < converters.size(); i++) {
		QDomElement converter = converters.at(i).toElement();

		if (converter.attribute("type") != "substitution") {
			continue;
		}

		converter.setAttribute("modes", 1);
		converter.setAttribute("substrings_map", "");
		converter.setAttribute("invert_order", false);
	}

	doc.documentElement().setAttribute("version", "9");
}

void PVRush::PVFormatVersion::__impl::from9to10(QDomDocument& doc)
{
	QDomNodeList axis = doc.documentElement().elementsByTagName("axis");
	for (int i = 0; i < axis.size(); i++) {
		QDomElement ax = axis.at(i).toElement();
		// Limit the choice of port plotting to uint16
		QDomElement plotted = ax.namedItem("plotting").toElement();
		QString plotting = plotted.attribute("mode");
		if (plotting == "port") {
			ax.setAttribute("type", "number_uint16");
		}
	}
	doc.documentElement().setAttribute("version", "10");
}

void PVRush::PVFormatVersion::__impl::from10to11(QDomDocument& doc)
{
	QDomNodeList axis = doc.documentElement().elementsByTagName("axis");
	for (int i = 0; i < axis.size(); i++) {
		QDomElement ax = axis.at(i).toElement();
		QDomElement plotting = ax.namedItem(PVFORMAT_XML_TAG_PLOTTING).toElement();
		if (plotting != QDomElement()) {
			plotting.setTagName(PVFORMAT_XML_TAG_SCALING);
		}
	}
	doc.documentElement().setAttribute("version", "11");
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
			QDomElement mapping_node = node.ownerDocument().createElement("mapping");
			mapping_node.setAttribute("mode", mapping);
			elt.appendChild(mapping_node);
			elt.removeAttribute("mapping");

			// Move plotting from attribute to node
			QString plotting = elt.attribute("plotting", QString());
			QDomElement plotting_node = node.ownerDocument().createElement("plotting");
			plotting_node.setAttribute("mode", plotting);
			elt.appendChild(plotting_node);
			elt.removeAttribute("plotting");

			// Move axis/time-format to axis/mapping/type_format
			QString tf = elt.attribute("time-format", QString());
			if (tf.size() > 0) {
				QDomElement axis_node = node.toElement();
				axis_node.setAttribute("type_format", tf);
			}
			elt.removeAttribute("time-format");
		}
	}

	QDomNode child = node.firstChild();
	while (!child.isNull()) {
		_rec_5to6(child);
		child = child.nextSibling();
	}
}
