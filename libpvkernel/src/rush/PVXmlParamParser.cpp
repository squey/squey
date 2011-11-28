/* 
 * File:   PVRush::PVXmlParamParser.cpp
 * Author: rpernaudat
 * 
 * Created on 12 mai 2011, 11:27
 */

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVFormatVersion.h>
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
		PVLOG_ERROR("(PVRush::PVXmlParamparser::PVXmlParamParser) file to parse not found!\n");
		return;
	}
	if (!xmlfile.open(QIODevice::ReadOnly | QIODevice::Text)) {
		PVLOG_ERROR("(PVRush::PVXmlParamparser::PVXmlParamParser) can't open file to parse.\n");
		return;
	}
	QTextStream tmpTextXml(&xmlfile); // file stream creation
	QDomDocument docXml;
	docXml.setContent(tmpTextXml.readAll());
	PVRush::PVFormatVersion::to_current(docXml);
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
	setDom(rootNode);
	setAxesCombinationFromRootNode(rootNode);
	dump_filters();
}

void PVRush::PVXmlParamParser::setVersionFromRootNode(QDomElement const& node)
{
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
		if (elt.tagName() != PVFORMAT_XML_TAG_AXES_COMBINATION_STR) {
			continue;
		}

		setAxesCombinationFromString(node.text());
		return;
	}
	setAxesCombinationFromString(QString());
}

void PVRush::PVXmlParamParser::setAxesCombinationFromString(QString const& str)
{
	PVLOG_DEBUG("(PVXmlParamParser::setAxesCombinationFromString) string: %s\n", qPrintable(str));
	_axes_combination.clear();
	if (str.isEmpty()) {
		return; // The default combination will be used
	}

	QStringList axes_list = str.split(',');
	PVCol naxes = _axes.size();
	_axes_combination.reserve(axes_list.size());
	for (int i = 0; i < axes_list.size(); i++) {
		bool ok = false;
		PVCol ax_id = axes_list[i].toLongLong(&ok);
		if (!ok || ax_id >= naxes) {
			continue;
		}
		_axes_combination.push_back(ax_id);
	}
}

QHash<int, QStringList> const& PVRush::PVXmlParamParser::getTimeFormat() const
{
	return time_format;
}

PVCore::PVArgumentList PVRush::PVXmlParamParser::getMapPlotParameters(QDomElement& elt, QString const& tag)
{
	PVCore::PVArgumentList args;
	QDomNodeList list = elt.elementsByTagName(tag);
	if (list.size() < 1) {
		return args;
	}
	
	QDomElement node = list.at(0).toElement();
	PVRush::PVXmlTreeNodeDom::toArgumentList(node, args);
	args.remove(PVFORMAT_MAP_PLOT_MODE_STR);
	return args;
}

int PVRush::PVXmlParamParser::setDom(QDomElement const& node, int id, QVector<uint32_t> tree_ids)
{
	int newId = id;
	if (id == -1) {
		return setDom(node, 0, tree_ids);
	}

	QDomNodeList childs = node.childNodes();
	int nchilds = childs.size();

	// For a given axis, we need to first create the corresponding one_to_one field filters
	// because we have the good id. Then we process the axes and the splitters that will change the next field's id !
	// The order of the different for loops is *important* here. Change it if you know what you're doing !
	for(int i=0; i< nchilds; i++) {
		QDomElement child = childs.at(i).toElement();
		QString node_type = getNodeType(child);

		if (node_type == PVFORMAT_XML_TAG_FILTER_STR) {
			pushFilter(child, newId);
		}
	}

	uint32_t* new_tree_id = NULL;
	for(int i=0; i< nchilds; i++) {
		QDomElement child = childs.at(i).toElement();
		QString node_type = getNodeType(child);
		if (node_type == PVFORMAT_XML_TAG_SPLITTER_STR) {
			pushFilter(child, newId);
			if (new_tree_id == NULL) {
				tree_ids.push_back(i);
				new_tree_id = &tree_ids.back();
			}
			else {
				*new_tree_id = i;
			}
			newId = setDom(child, newId, tree_ids);
		}
		else
		if (node_type == PVFORMAT_XML_TAG_FIELD_STR) {
			if (new_tree_id == NULL) {
				tree_ids.push_back(i);
				new_tree_id = &tree_ids.back();
			}
			else {
				*new_tree_id = i;
			}
			newId = setDom(child, newId, tree_ids);
		}
	}

	for(size_t i = 0; i < nchilds; i++){
		QDomElement child = childs.at(i).toElement();

		if (getNodeType(child) == "axis") {
			PVAxisFormat axis;
			axis.set_name(child.attribute(PVFORMAT_AXIS_NAME_STR, PVFORMAT_AXIS_NAME_DEFAULT));
			axis.set_type(child.attribute(PVFORMAT_AXIS_TYPE_STR, PVFORMAT_AXIS_TYPE_DEFAULT));
			axis.set_mapping(child.attribute(PVFORMAT_AXIS_MAPPING_STR, PVFORMAT_AXIS_MAPPING_DEFAULT));
			axis.set_plotting(child.attribute(PVFORMAT_AXIS_PLOTTING_STR, PVFORMAT_AXIS_PLOTTING_DEFAULT));
			axis.set_group(child.attribute(PVFORMAT_AXIS_GROUP_STR, PVFORMAT_AXIS_GROUP_DEFAULT));
			axis.set_color(child.attribute(PVFORMAT_AXIS_COLOR_STR, PVFORMAT_AXIS_COLOR_DEFAULT));
			axis.set_titlecolor(child.attribute(PVFORMAT_AXIS_TITLECOLOR_STR, PVFORMAT_AXIS_TITLECOLOR_DEFAULT));
			axis.compute_unique_id(tree_ids);
			QString tag = child.attribute(PVFORMAT_AXIS_TAG_STR, QString());
			if (!tag.isEmpty()) {
				QStringList tags = tag.split(PVFORMAT_TAGS_SEP);
				for (int j = 0; j < tags.size(); j++) {
					axis.add_tag(tags[j]);
				}
			}
			if(child.attribute(PVFORMAT_AXIS_TYPE_STR, PVFORMAT_AXIS_TYPE_DEFAULT)=="time"){
				time_format[newId+1]  = child.attribute(PVFORMAT_AXIS_TIMEFORMAT_STR, PVFORMAT_AXIS_TIMEFORMAT_DEFAULT).split("\n");
			}

			// Mapping and plotting parameters
			PVLOG_INFO("For axis %s:\n", qPrintable(axis.get_name()));
			PVCore::PVArgumentList args = getMapPlotParameters(child, PVFORMAT_XML_TAG_MAPPING);
			axis.set_args_mapping(args);
			PVLOG_INFO("Mapping args:\n");
			PVCore::PVArgumentList::const_iterator it;
			for (it = args.begin(); it != args.end(); it++) {
				PVLOG_INFO("key: %s | value: %s\n", qPrintable(it.key()), qPrintable(it.value().toString()));
			}
			PVLOG_INFO("Plotting args:\n");
			args = getMapPlotParameters(child, PVFORMAT_XML_TAG_PLOTTING);
			for (it = args.begin(); it != args.end(); it++) {
				PVLOG_INFO("key: %s | value: %s\n", qPrintable(it.key()), qPrintable(it.value().toString()));
			}
			axis.set_args_plotting(args);
			_axes.push_back(axis);

			newId++;

			// Only one axis per field
			break;
		}
	}

	return newId;
}

void PVRush::PVXmlParamParser::pushFilter(QDomElement const& elt, int newId)
{
	static PVCore::PVClassLibrary<PVFilter::PVFieldsFilterReg> const& filters_lib = PVCore::PVClassLibrary<PVFilter::PVFieldsFilterReg>::get();
	QString node_type = getNodeType(elt);
	QString filter_plugin_name = elt.attribute(PVFORMAT_FILTER_TYPE_STR, PVFORMAT_FILTER_TYPE_DEFAULT);
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
	PVCol axis_id = 0;
	for (int i = 0; i < children.size(); i++) {
		QDomElement elt_child = children.at(i).toElement();
		if (elt_child.tagName() != PVFORMAT_XML_TAG_FIELD_STR) {
			continue;
		}

		// The axis is one of its children
		QDomNodeList field_children = elt_child.childNodes();
		QDomElement axis_child;
		bool found = false;
		for (int i = 0; i < field_children.size(); i++) {
			axis_child = field_children.at(i).toElement();
			if (axis_child.tagName() == PVFORMAT_XML_TAG_AXIS_STR) {
				found = true;
				break;
			}
		}
		if (found) {
			QString tag = axis_child.attribute(PVFORMAT_AXIS_TAG_STR, PVFORMAT_AXIS_TAG_DEFAULT);
			if (!tag.isEmpty()) {
				QStringList tags = tag.split(PVFORMAT_TAGS_SEP);
				for (int i = 0; i < tags.size(); i++) {
					axes[tags[i]] = axis_id;
				}
			}
		}
		axis_id++;
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

