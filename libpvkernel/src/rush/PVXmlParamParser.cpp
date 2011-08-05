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
	setVersionFromRootNode(docXml.documentElement());
	setDom(docXml.documentElement());

	xmlfile.close();

	dump_filters();
}

void PVRush::PVXmlParamParser::dump_filters()
{
	QList<PVXmlParamParserData>::iterator it;
	for (it = fields.begin(); it != fields.end(); it++) {
		PVLOG_DEBUG("On axis %d, apply filter %s\n", it->axis_id, qPrintable(it->filter_lib->registered_name()));
	}
}

PVRush::PVXmlParamParser::PVXmlParamParser(QDomElement const& rootNode)
{
	setVersionFromRootNode(rootNode);
	setDom(rootNode);
}

void PVRush::PVXmlParamParser::setVersionFromRootNode(QDomElement const& node)
{
	format_version = node.attribute("version","0").toInt();
}

PVRush::PVXmlParamParser::~PVXmlParamParser()
{
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
		newId = 0;
		if (node.firstChild().toElement().tagName() == "field") {
			newId = setDom(node.firstChild().toElement(), 0);
		} else {
			setDom(node, 0);
		}
	}
	else {
		size_t nchilds = countChild(node.toElement());

		switch (format_version) {
			case 0:
			{
				// For a given axis, we need to first creatte the corresponding one_to_one field filters
				// because we have good id. Then we process the axes and the splitters that will change the next field id !
				// The order of the different for loops is *important* here. Change it if you know what you're doing !
				for(size_t i = 0; i < nchilds; i++){
					QDomElement child(node.childNodes().at(i).toElement());

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
					QDomElement child(node.childNodes().at(i).toElement());
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

					// Process children
					for (int iF = 0; iF < countChild(child.toElement()); iF++) {
						newId = setDom(child.childNodes().at(iF).toElement(), newId);
					}
				}

				break;
			}
			case 1:
			{
				// Same as above, twoseparate for loops because the order of processing is important !
				// Change this if you know what you're doing !!
				for(int i=0; i< countChild(node.toElement()); i++) {
					QDomElement child(node.childNodes().at(i).toElement());
					QString node_type = getNodeType(child);

					if (node_type == "filter") {
						pushFilter(child, newId);
					}
				}

				for(int i=0; i< countChild(node.toElement()); i++) {
					QDomElement child(node.childNodes().at(i).toElement());
					QString node_type = getNodeType(child);
					if (node_type == "splitter") {
						pushFilter(child, newId);

						// Process children
						for (int iF = 0; iF < countChild(child.toElement()); iF++) {
							newId = setDom(child.childNodes().at(iF).toElement(), newId);
						}
					}
				}

				break;
			}
			default:
			{
			}
		}

		for(size_t i = 0; i < nchilds; i++){
			QDomElement child(node.childNodes().at(i).toElement());

			if (getNodeType(child) == "axis") {
				QHash<QString,QString> hash;
				hash.insert("name",child.attribute("name",""));
				hash.insert("type",child.attribute("type",""));
				hash.insert("mapping",child.attribute("mapping",""));
				hash.insert("plotting",child.attribute("plotting",""));
				hash.insert("key",child.attribute("key",""));
				hash.insert("group",child.attribute("group",""));
				hash.insert("color",child.attribute("color",""));
				hash.insert("titlecolor",child.attribute("titlecolor",""));
				axes.push_back(hash);
				if(child.attribute("type","")=="time"){
					PVLOG_DEBUG("Time format for axis %d\n", newId);
					time_format[newId+1]  = child.attribute("time-format","").split("\n");
				}

				newId++;
			}
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
	data.nchildren = elt.childNodes().size();
	PVRush::PVXmlTreeNodeDom tnd(elt);
	tnd.toArgumentList(data.filter_lib->get_default_args(), data.filter_args);
	fields.push_back(data);
}

QList<QHash<QString, QString> > const& PVRush::PVXmlParamParser::getAxes() const
{
	return this->axes;
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

