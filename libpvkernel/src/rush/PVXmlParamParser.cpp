/* 
 * File:   PVRush::PVXmlParamParser.cpp
 * Author: rpernaudat
 * 
 * Created on 12 mai 2011, 11:27
 */

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVXmlTreeNodeDom.h>
#include <pvkernel/rush/PVXmlParamParser.h>

// Exceptions 

PVRush::PVXmlParamParserExceptionPluginNotFound::PVXmlParamParserExceptionPluginNotFound(QString type, QString plugin_name)
{
	_what = QString("Plugin '") + plugin_name + QString("' of type '") + type + QString("' isn't available.");
}

QString PVRush::PVXmlParamParserExceptionPluginNotFound::what()
{
	return _what;
}

// PVXmlParamParser class

PVRush::PVXmlParamParser::PVXmlParamParser(QString const& nameFile)
{
	QFile xmlfile(nameFile);

	if(!xmlfile.exists()) {
		PVLOG_ERROR("File to parse not found!\n");
		return;
	}
	if (!xmlfile.open(QIODevice::ReadOnly | QIODevice::Text)) {
		PVLOG_ERROR("Can't open file to parse.\n");
		return;
	}
	QTextStream tmpTextXml(&xmlfile); // file stream creation
	QDomDocument docXml;
	docXml.setContent(tmpTextXml.readAll());
	parseFromRootNode(docXml.documentElement());

	xmlfile.close();
}

PVRush::PVXmlParamParser::PVXmlParamParser(QDomElement const& rootNode)
{
	parseFromRootNode(rootNode);
}

PVRush::PVXmlParamParser::~PVXmlParamParser()
{
}

void PVRush::PVXmlParamParser::dump_filters()
{
	QList<PVXmlParamParserData>::iterator it;
	for (it = fields.begin(); it != fields.end(); it++) {
		PVLOG_DEBUG("On axis %d, apply filter %s\n", it->axis_id, qPrintable(it->filter_lib->registered_name()));
	}
}

void PVRush::PVXmlParamParser::parseFromRootNode(QDomElement const& rootNode)
{
	setVersionFromRootNode(rootNode);
	setDom(rootNode);
	setAxesCombinationFromRootNode(rootNode);
	dump_filters();
}

void PVRush::PVXmlParamParser::setVersionFromRootNode(QDomElement const& node)
{
	format_version = node.attribute("version","0").toInt();
}

void PVRush::PVXmlParamParser::setAxesCombinationFromRootNode(QDomElement const& node)
{
	QDomNodeList childs = node.childNodes();
	for (int i = 0; i < childs.size(); i++) {
		QDomNode child_node = childs.at(i);
		if (!child_node.isElement()) {
			continue;
		}
		
		QDomElement elt = child_node.toElement();
		if (elt.tagName() != PVFORMAT_AXES_COMBINATION) {
			continue;
		}

		setAxesCombinationFromString(node.text());
		return;
	}
	setAxesCombinationFromString(QString());
}

void PVRush::PVXmlParamParser::setAxesCombinationFromString(QString const& str)
{
	_axes_combination.clear();
	if (str.isEmpty()) {
		return; // The default combination will be used
	}

	QStringList axes_list = str.split(',');
	PVCol naxes = _axes.size();
	_axes_combination.resize(axes_list.size());
	for (int i = 0; i < axes_list.size(); i++) {
		bool ok = false;
		PVCol ax_id = axes_list[i].toLongLong(&ok);
		if (!ok || ax_id >= naxes || std::find(_axes_combination.begin(), _axes_combination.end(), ax_id) != _axes_combination.end()) {
			continue;
		}
		_axes_combination.push_back(ax_id);
	}
}

QHash<int, QStringList> const& PVRush::PVXmlParamParser::getTimeFormat() const
{
	return time_format;
}

int PVRush::PVXmlParamParser::setDom(QDomElement const& node, int id)
{
	static PVCore::PVClassLibrary<PVFilter::PVFieldsFilterReg> const& filters_lib = PVCore::PVClassLibrary<PVFilter::PVFieldsFilterReg>::get();

	int newId = id;
	if (id == -1) {
		/*if (node.firstChild().toElement().tagName() == "field") {
			newId = setDom(node.firstChild().toElement(), 0);
		} else {
			setDom(node, 0);
		}*/
		return setDom(node, 0);
	}

	QDomNodeList childs = node.childNodes();
	int nchilds = childs.size();

	switch (format_version) {
		case 0:
		{
			// For a given axis, we need to first create the corresponding one_to_one field filters
			// because we have the good id. Then we process the axes and the splitters that will change the next field's id !
			// The order of the different for loops is *important* here. Change it if you know what you're doing !
			for(size_t i = 0; i < nchilds; i++){
				QDomElement child = childs.at(i).toElement();

				if(getNodeType(child)=="filter"){
					PVRush::PVXmlParamParserData data;
					data.axis_id = newId;
					data.filter_lib = filters_lib.get_class_by_name("filter_regexp");
					if (!data.filter_lib) {
						throw PVXmlParamParserExceptionPluginNotFound("filter", "regexp");
					}
					data.filter_args["regexp"] = getNodeRegExp(child.toElement());
					data.filter_args["reverse"] = getNodeTypeGrep(child.toElement()).compare("include") != 0;
					fields.push_back(data);
				}
			}

			for (size_t i = 0; i < nchilds; i++) {
				QDomElement child = childs.at(i).toElement();
				PVRush::PVXmlParamParserData data;
				data.nchildren = nchilds;

				if (getNodeType(child) == "RegEx") {
					data.axis_id = newId;
					data.filter_lib = filters_lib.get_class_by_name("splitter_regexp");
					if (!data.filter_lib) {
						throw PVXmlParamParserExceptionPluginNotFound("splitter", "regexp");
					}
					data.filter_args["regexp"] = getNodeRegExp(child.toElement());
					fields.push_back(data);
				}
				else
				if (getNodeType(child) == "url") {
					data.axis_id = newId;
					data.filter_lib = filters_lib.get_class_by_name("splitter_url");
					if (!data.filter_lib) {
						throw PVXmlParamParserExceptionPluginNotFound("splitter", "url");
					}
					fields.push_back(data);
				}
				else
				if (getNodeType(child) == "pcap") {
					data.axis_id = newId;
					data.filter_lib = filters_lib.get_class_by_name("splitter_pcap");
					if (!data.filter_lib) {
						throw PVXmlParamParserExceptionPluginNotFound("splitter", "pcap");
					}
					data.filter_args["datalink"] = QVariant((int) -1);
					fields.push_back(data);
				}
				else
				if (getNodeType(child) == "csv") {
					data.axis_id = newId;
					data.filter_lib = filters_lib.get_class_by_name("splitter_csv");
					if (!data.filter_lib) {
						throw PVXmlParamParserExceptionPluginNotFound("splitter", "csv");
					}
					data.filter_args["sep"] = child.attribute("delimiter","").at(0);
					fields.push_back(data);
				}

				// Process children of this child
				QDomNodeList node_childs = child.childNodes();
				int node_nchilds = node_childs.size();
				for (int iF = 0; iF < node_nchilds; iF++) {
					newId = setDom(node_childs.at(iF).toElement(), newId);
				}
			}

			break;
		}
		case 1:
		{
			// Same as above, two separate for loops because the order of processing is important !
			// Change this if you know what you're doing !!
			for(int i=0; i< nchilds; i++) {
				QDomElement child = childs.at(i).toElement();
				QString node_type = getNodeType(child);

				if (node_type == "filter") {
					pushFilter(child, newId);
				}
			}

			for(int i=0; i< nchilds; i++) {
				QDomElement child = childs.at(i).toElement();
				QString node_type = getNodeType(child);
				if (node_type == "splitter") {
					pushFilter(child, newId);
					newId = setDom(child, newId);
				}
				else
				if (node_type == "field") {
					newId = setDom(child, newId);
				}
			}

			break;
		}
		default:
		{
		}
	}

	for(size_t i = 0; i < nchilds; i++){
		QDomElement child = childs.at(i).toElement();

		if (getNodeType(child) == "axis") {
			PVAxisFormat axis;
			axis.set_name(child.attribute("name",""));
			axis.set_type(child.attribute("type",""));
			axis.set_mapping(child.attribute("mapping",""));
			axis.set_plotting(child.attribute("plotting",""));
			axis.set_key(child.attribute("key",""));
			axis.set_group(child.attribute("group",""));
			axis.set_color(child.attribute("color",""));
			axis.set_titlecolor(child.attribute("titlecolor",""));
			_axes.push_back(axis);
			if(child.attribute("type","")=="time"){
				time_format[newId+1]  = child.attribute("time-format","").split("\n");
			}

			newId++;
		}
	}

	return newId;
}

void PVRush::PVXmlParamParser::pushFilter(QDomElement const& elt, int newId)
{
	static PVCore::PVClassLibrary<PVFilter::PVFieldsFilterReg> const& filters_lib = PVCore::PVClassLibrary<PVFilter::PVFieldsFilterReg>::get();
	QString node_type = getNodeType(elt);
	QString filter_plugin_name = elt.attribute("type","");
	PVCore::PVArgumentList args;
	PVRush::PVXmlParamParserData data;
	data.axis_id = newId;
	data.filter_lib = filters_lib.get_class_by_name(node_type + QString("_") + filter_plugin_name);
	if (!data.filter_lib) {
		throw PVXmlParamParserExceptionPluginNotFound(node_type, filter_plugin_name);
	}
	// Get the list of the filter axes' tags and pass this to the filter
	QDomNodeList children = elt.childNodes();
	PVFilter::filter_child_axes_tag_t& axes(data.children_axes_tag);
	axes.reserve(children.size());
	for (int i = 0; i < children.size(); i++) {
		QDomElement elt_child = children.at(i).toElement();
		if (elt_child.tagName() != "field") {
			continue;
		}

		// The axis is one of its children
		QDomNodeList field_children = elt_child.childNodes();
		QDomElement axis_child;
		bool found = false;
		for (int i = 0; i < field_children.size(); i++) {
			axis_child = field_children.at(i).toElement();
			if (axis_child.tagName() == "axis") {
				found = true;
				break;
			}
		}
		if (!found) {
			continue;
		}

		// TODO: change "name" by "tag" when that will be done !
		QString tag = axis_child.attribute("name", "");
		if (!tag.isEmpty()) {
			axes.push_back(tag);
		}
	}
	data.nchildren = elt.childNodes().size();
	PVRush::PVXmlTreeNodeDom tnd(elt);
	tnd.toArgumentList(data.filter_lib->get_default_args(), data.filter_args);
	fields.push_back(data);
}

PVRush::list_axes_t const& PVRush::PVXmlParamParser::getAxes() const
{
	return _axes;
}


/******************************* private *************************************/
int PVRush::PVXmlParamParser::countChild(QDomElement node)
{
	return node.childNodes().count();
}

QList<PVRush::PVXmlParamParserData> const& PVRush::PVXmlParamParser::getFields() const
{
	return fields;
}

QString PVRush::PVXmlParamParser::getNodeRegExp(QDomElement node)
{
	return node.attribute("expression","");
}

QString PVRush::PVXmlParamParser::getNodeTypeGrep(QDomElement node)
{
	return node.attribute("type","");
}

QString PVRush::PVXmlParamParser::getNodeName(QDomElement node)
{
	return node.attribute("name","");
}

QString PVRush::PVXmlParamParser::getNodeType(QDomElement node)
{
	return node.tagName();
}

