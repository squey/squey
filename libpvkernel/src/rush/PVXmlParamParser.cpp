/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

/*
 * File:   PVRush::PVXmlParamParser.cpp
 * Author: rpernaudat
 *
 * Created on 12 mai 2011, 11:27
 */

#include <pvkernel/rush/PVAxisFormat.h>    // for PVAxisFormat, etc
#include <pvkernel/rush/PVFormatVersion.h> // for to_current
#include <pvkernel/rush/PVFormat_types.h>
#include <pvkernel/rush/PVXmlParamParser.h> // for PVXmlParamParser, etc
#include <pvkernel/rush/PVXmlParamParserData.h>
#include <pvkernel/rush/PVXmlTreeNodeDom.h> // for PVXmlTreeNodeDom
#include <pvkernel/rush/PVUnicodeSource.h>

#include <pvkernel/filter/PVFieldsFilter.h> // for PVFieldsFilterReg_p, etc

#include <pvkernel/core/PVArgument.h>     // for PVArgumentList
#include <pvkernel/core/PVClassLibrary.h> // for PVClassLibrary
#include <pvkernel/core/PVUtils.h>
#include <pvkernel/core/PVLogger.h>       // for PVLOG_DEBUG, PVLOG_ERROR

#include <pvbase/types.h> // for PVCol

#include <QDomElement>
#include <QFile> // for QFile
#include <QIODevice>
#include <QList>       // for QList, QList<>::iterator
#include <QString>     // for QString, operator+, etc
#include <QStringList> // for QStringList
#include <QTextStream>

#include <memory>    // for __shared_ptr
#include <numeric>   // for iota
#include <stdexcept> // for runtime_error
#include <vector>    // for vector

// PVXmlParamParser class

PVRush::PVXmlParamParser::PVXmlParamParser(QString const& nameFile,
                                           bool add_input_column_name /* = false */)
{
	QFile xmlfile(nameFile);

	if (!xmlfile.exists() or !xmlfile.open(QIODevice::ReadOnly | QIODevice::Text)) {
		PVLOG_ERROR("(PVRush::PVXmlParamparser::PVXmlParamParser) file to parse not found!\n");
		throw PVInvalidFile("Can't open file to parse : " + nameFile.toStdString());
	}
	QTextStream tmpTextXml(&xmlfile); // file stream creation
	QDomDocument docXml;
	QString error;
	int line, col;
	if (!docXml.setContent(tmpTextXml.readAll(), &error, &line, &col)) {
		throw PVInvalidFile((error + " at line : " + QString::number(line) + " col:" +
		                     QString::number(col) + " in file : " + nameFile)
		                        .toStdString());
	}
	PVRush::PVFormatVersion::to_current(docXml);
	if (add_input_column_name) {
		addInputNameColumn(docXml);
	}
	parseFromRootNode(docXml.documentElement());

	xmlfile.close();
}

void PVRush::PVXmlParamParser::addInputNameColumn(QDomDocument& xml)
{
	// Create CSV splitter for input names column
	QDomElement csv_splitter_dom = xml.createElement(PVFORMAT_XML_TAG_SPLITTER_STR);
	csv_splitter_dom.setAttribute(PVFORMAT_FILTER_TYPE_STR, "csv");
	csv_splitter_dom.setAttribute("quote", "\"");
	csv_splitter_dom.setAttribute("sep",
	                              QString(PVRush::PVUnicodeSource<>::MULTI_INPUTS_SEPARATOR));

	QDomElement field = xml.createElement(PVFORMAT_XML_TAG_FIELD_STR);
	csv_splitter_dom.appendChild(field);
	QDomElement axis = xml.createElement(PVFORMAT_XML_TAG_AXIS_STR);

	QDomElement mapping = xml.createElement(PVFORMAT_XML_TAG_MAPPING);
	mapping.setAttribute(PVFORMAT_MAP_PLOT_MODE_STR, PVFORMAT_AXIS_MAPPING_DEFAULT);
	QDomElement plotting = xml.createElement(PVFORMAT_XML_TAG_PLOTTING);
	plotting.setAttribute(PVFORMAT_MAP_PLOT_MODE_STR, PVFORMAT_AXIS_PLOTTING_DEFAULT);
	axis.appendChild(mapping);
	axis.appendChild(plotting);

	axis.setAttribute(PVFORMAT_AXIS_NAME_STR, "file_name");
	axis.setAttribute(PVFORMAT_AXIS_TYPE_STR, "string");
	field.appendChild(axis);

	QDomElement field2 = xml.createElement(PVFORMAT_XML_TAG_FIELD_STR);
	csv_splitter_dom.appendChild(field2);

	QDomNode root_splitter = xml.documentElement().firstChildElement(PVFORMAT_XML_TAG_SPLITTER_STR);
	root_splitter.parentNode().removeChild(root_splitter);
	QDomNode param = xml.documentElement();
	param.appendChild(csv_splitter_dom);
	field2.appendChild(root_splitter);

	// Recompute axis combination to insert "input_name" as first axis
	QDomNode axes_combination = xml.documentElement().firstChildElement("axes-combination");
	axes_combination.parentNode().removeChild(axes_combination);
	QStringList comb = axes_combination.toElement().text().split(",");
	for (QString& str : comb) {
		str = QString::number(str.toUInt() + 1);
	}
	comb.insert(0, QString::number(0));
	QString new_axes_comb = comb.join(",");
	QDomElement ac = xml.createElement(PVFORMAT_XML_TAG_AXES_COMBINATION_STR);
	QDomText new_axes_comb_text_node = xml.createTextNode(new_axes_comb);
	ac.appendChild(new_axes_comb_text_node);
	xml.documentElement().appendChild(ac);
}

PVRush::PVXmlParamParser::PVXmlParamParser(QDomElement const& rootNode)
{
	parseFromRootNode(rootNode);
}

PVRush::PVXmlParamParser::~PVXmlParamParser() = default;

void PVRush::PVXmlParamParser::dump_filters()
{
	QList<PVXmlParamParserData>::iterator it;
	for (it = fields.begin(); it != fields.end(); it++) {
		PVLOG_DEBUG("On axis %d, apply filter %s\n", it->axis_id,
		            qPrintable(it->filter_lib->registered_name()));
	}
}

void PVRush::PVXmlParamParser::parseFromRootNode(QDomElement const& rootNode)
{
	setDom(rootNode);
	setAxesCombinationFromRootNode(rootNode);
	setLinesRangeFromRootNode(rootNode);
	setPythonScriptFromRootNode(rootNode);
	dump_filters();
}

void PVRush::PVXmlParamParser::setLinesRangeFromRootNode(QDomElement const& rootNode)
{
	_first_line = rootNode.attribute("first_line", "0").toULongLong();
	_line_count = rootNode.attribute("line_count", "0").toULongLong();
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

		setAxesCombinationFromString(elt.text());
		return;
	}
	setAxesCombinationFromString(QString());
}

void PVRush::PVXmlParamParser::setAxesCombinationFromString(QString const& str)
{
	PVLOG_DEBUG("(PVXmlParamParser::setAxesCombinationFromString) string: %s\n", qPrintable(str));
	_axes_combination.clear();

	QStringList axes_list = str.split(',');
	PVCol naxes = PVCol(_axes.size());

	if (axes_list.size() < 2) {
		_axes_combination.resize(naxes);
		std::iota(_axes_combination.begin(), _axes_combination.end(), PVCol(0));
		return; // The default combination will be used
	}

	_axes_combination.reserve(axes_list.size());
	for (int i = 0; i < axes_list.size(); i++) {
		bool ok = false;
		PVCol ax_id = PVCol(axes_list[i].toLongLong(&ok));
		if (!ok || ax_id >= naxes) {
			continue;
		}
		_axes_combination.push_back(ax_id);
	}
}

void PVRush::PVXmlParamParser::setPythonScriptFromRootNode(QDomElement const& rootNode)
{
	QDomElement python_script_element = rootNode.firstChildElement("python-script");

	QString python_script = python_script_element.text();
	bool is_path = python_script_element.attribute("path", "1").toUInt();
	bool disabled = python_script_element.attribute("disabled", "0").toUInt();
	if (not is_path) {
		python_script = PVCore::deserialize_base64<QString>(python_script);
	}
	setPythonScriptFromFile(python_script, is_path, disabled);
}	

void PVRush::PVXmlParamParser::setPythonScriptFromFile(QString const& python_script, bool as_path, bool disabled)
{
	_python_script = python_script;
	_python_script_is_path = as_path;
	_python_script_disabled = disabled;
}

PVRush::PVAxisFormat::node_args_t
PVRush::PVXmlParamParser::getMapPlotParameters(QDomElement& elt, QString const& tag, QString& mode)
{
	PVAxisFormat::node_args_t args;
	// Get the first mapping/plotting elements (ignore other if there are multiples)
	QDomElement node = elt.firstChildElement(tag);
	if (node.isNull()) {
		// No mapping or plotting, use default mapping/plotting
		mode = PVFORMAT_MAP_PLOT_MODE_DEFAULT;
		return {};
	}

	// Save every attributes
	QDomNamedNodeMap attrs = node.attributes();
	for (int i = 0; i < attrs.size(); i++) {
		QDomAttr a = attrs.item(i).toAttr();
		args[a.name()] = a.value();
	}

	// Define mode.
	if (args.contains(PVFORMAT_MAP_PLOT_MODE_STR)) {
		mode = args.take(PVFORMAT_MAP_PLOT_MODE_STR);
	} else {
		mode = PVFORMAT_MAP_PLOT_MODE_DEFAULT;
	}

	return args;
}

int PVRush::PVXmlParamParser::setDom(QDomElement const& node, int id)
{
	int newId = id;
	if (id == -1) {
		return setDom(node, 0);
	}

	QDomNodeList childs = node.childNodes();
	int nchilds = childs.size();

	// For a given axis, we need to first create the corresponding one_to_one field filters
	// because we have the good id. Then we process the axes and the splitters that will change the
	// next field's id !
	// The order of the different for loops is *important* here. Change it if you know what you're
	// doing !
	for (int i = 0; i < nchilds; i++) {
		QDomElement child = childs.at(i).toElement();
		QString node_type = getNodeType(child);

		if (node_type == PVFORMAT_XML_TAG_FILTER_STR ||
		    node_type == PVFORMAT_XML_TAG_CONVERTER_STR) {
			pushFilter(child, newId);
		}
	}

	for (int i = 0; i < nchilds; i++) {
		QDomElement child = childs.at(i).toElement();
		QString node_type = getNodeType(child);
		if (node_type == PVFORMAT_XML_TAG_SPLITTER_STR) {
			pushFilter(child, newId);
			newId = setDom(child, newId);
		} else if (node_type == PVFORMAT_XML_TAG_FIELD_STR) {
			if (child.hasChildNodes()) {
				QDomNodeList nodes = child.childNodes();
				bool has_axis = false;
				for (int i = 0; i < nodes.size(); ++i) {
					if (getNodeType(nodes.at(i).toElement()) == PVFORMAT_XML_TAG_AXIS_STR) {
						has_axis = true;
						break;
					}
				}

				if (has_axis) {
					_fields_mask.push_back(true);
				}
			} else {
				_fields_mask.push_back(false);
				++newId;
			}

			newId = setDom(child, newId);
		}
	}

	for (int i = 0; i < nchilds; i++) {
		QDomElement child = childs.at(i).toElement();

		if (getNodeType(child) == PVFORMAT_XML_TAG_AXIS_STR) {
			PVAxisFormat axis(PVCol(_axes.size()));
			axis.set_name(child.attribute(PVFORMAT_AXIS_NAME_STR, PVFORMAT_AXIS_NAME_DEFAULT));
			axis.set_type(child.attribute(PVFORMAT_AXIS_TYPE_STR, PVFORMAT_AXIS_TYPE_DEFAULT));
			axis.set_type_format(
			    child.attribute(PVFORMAT_AXIS_TYPE_FORMAT_STR, PVFORMAT_AXIS_TYPE_FORMAT_DEFAULT));
			axis.set_color(child.attribute(PVFORMAT_AXIS_COLOR_STR, PVFORMAT_AXIS_COLOR_DEFAULT));
			axis.set_titlecolor(
			    child.attribute(PVFORMAT_AXIS_TITLECOLOR_STR, PVFORMAT_AXIS_TITLECOLOR_DEFAULT));

			// Mapping and plotting parameters
			QString mode;
			PVAxisFormat::node_args_t args =
			    getMapPlotParameters(child, PVFORMAT_XML_TAG_MAPPING, mode);
			axis.set_mapping(mode);
			axis.set_args_mapping(args);
			args = getMapPlotParameters(child, PVFORMAT_XML_TAG_PLOTTING, mode);
			axis.set_plotting(mode);
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
	static PVCore::PVClassLibrary<PVFilter::PVFieldsFilterReg> const& filters_lib =
	    PVCore::PVClassLibrary<PVFilter::PVFieldsFilterReg>::get();
	QString node_type = getNodeType(elt);
	QString filter_plugin_name =
	    elt.attribute(PVFORMAT_FILTER_TYPE_STR, PVFORMAT_FILTER_TYPE_DEFAULT);
	PVCore::PVArgumentList args;
	PVRush::PVXmlParamParserData data;
	data.axis_id = newId;
	data.filter_lib = filters_lib.get_class_by_name(node_type + QString("_") + filter_plugin_name);
	data.nchildren = elt.childNodes().size();
	PVRush::PVXmlTreeNodeDom tnd(elt);
	tnd.toArgumentList(data.filter_lib->get_default_args(), data.filter_args);
	fields.push_back(data);
}

QList<PVRush::PVAxisFormat> const& PVRush::PVXmlParamParser::getAxes() const
{
	return _axes;
}

QString PVRush::PVXmlParamParser::get_python_script(bool& as_path, bool& disabled) const
{
	as_path = _python_script_is_path;
	disabled = _python_script_disabled;
	return _python_script;
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
	return node.attribute("expression", "");
}

QString PVRush::PVXmlParamParser::getNodeTypeGrep(QDomElement node)
{
	return node.attribute("type", "");
}

QString PVRush::PVXmlParamParser::getNodeName(QDomElement node)
{
	return node.attribute("name", "");
}

QString PVRush::PVXmlParamParser::getNodeType(QDomElement node)
{
	return node.tagName();
}
