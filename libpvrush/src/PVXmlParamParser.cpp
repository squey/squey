/* 
 * File:   PVRush::PVXmlParamParser.cpp
 * Author: rpernaudat
 * 
 * Created on 12 mai 2011, 11:27
 */

#include "pvrush/PVXmlParamParser.h"

PVRush::PVXmlParamParser::PVXmlParamParser(QString nameFile)
{
	QFile fichier(nameFile);
	if(!fichier.exists()) {
		//le fichier n'existe pas.
		PVLOG_ERROR("File to parse unfound!\n");
		return;
	}
	if (!fichier.open(QIODevice::ReadOnly | QIODevice::Text)) {
		PVLOG_ERROR("Can't open file to parse.\n");
		return;
	}
	QTextStream tmpTextXml(&fichier);//creation of the file stream
	QDomDocument docXml;
	docXml.setContent(tmpTextXml.readAll());
	format_version = docXml.documentElement().attribute("version","0").toInt();
	setDom(docXml.documentElement());

	fichier.close();
}

PVRush::PVXmlParamParser::~PVXmlParamParser()
{
}

QHash<int, QStringList> const& PVRush::PVXmlParamParser::getTimeFormat() const
{
	return time_format;
}

int PVRush::PVXmlParamParser::setDom(QDomElement node, int id)
{
	int newId = id;
	if (id == -1) {
		newId = 0;
		if (node.firstChild().toElement().tagName() == "field") {
			newId = setDom(node.firstChild().toElement(), 0);
		} else {
			setDom(node, 0);
		}
	}
	else
	switch (format_version) {
		case 0:
		{
			bool process_childs = false;

			for(int i=0;i<countChild(node.toElement());i++){
				QDomElement child(node.childNodes().at(i).toElement());

				if(getNodeType(child)=="filter"){
					PVRush::PVXmlParamParserData data;
					data.axis_id = newId;
					data.filter_type = "filter";
					data.filter_plugin_name = "regexp";
					data.filter_args["regexp"] = getNodeRegExp(child.toElement());
					data.filter_args["reverse"] = getNodeTypeGrep(child.toElement()).compare("include") == 0;
					fields.push_back(data);
				}
				else
				if (getNodeType(child) == "RegEx") {
					PVRush::PVXmlParamParserData data;
					data.axis_id = newId;
					data.filter_type = "splitter";
					data.filter_plugin_name = "regexp";
					data.filter_args["regexp"] = getNodeRegExp(child.toElement());
					fields.push_back(data);

					process_childs = true;
				}
				else
				if (getNodeType(child) == "axis") {
					QHash<QString,QString> hash;//=new QHash<QString,QString>();
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
				else
				if (getNodeType(child) == "url") {
					PVRush::PVXmlParamParserData data;
					data.axis_id = newId;
					data.filter_type = "splitter";
					data.filter_plugin_name = "url";
					fields.push_back(data);
					process_childs = true;
				}
				else
				if (getNodeType(child) == "pcap") {
					PVRush::PVXmlParamParserData data;
					data.axis_id = newId;
					data.filter_type = "splitter";
					data.filter_plugin_name = "pcap";
					fields.push_back(data);
					process_childs = true;
				}
				else
				if (getNodeType(child) == "csv") {
					PVRush::PVXmlParamParserData data;
					data.axis_id = newId;
					data.filter_type = "splitter";
					data.filter_plugin_name = "csv";
					data.filter_args["sep"] = child.attribute("delimiter","").at(0);
					fields.push_back(data);
					process_childs = true;
				}

				if (process_childs) {
					for (int iF = 0; iF < countChild(child.toElement()); iF++) {
						newId = setDom(child.childNodes().at(iF).toElement(), newId);
					}
				}
			}
			break;
		}
		case 1:
		{
//			bool process_childs = false;
//
//			for(int i=0; i< countChild(node.toElement()); i++) {
//				QDomElement child(node.childNodes().at(i).toElement());
//				QString node_type = getNodeType(child);
//
//				if (node_type == "splitter") {
//					PVArgumentList args;
//					PVRush::PVXmlParamParserData data;
//					data.axis_id = newId;
//					data.filter_type = "splitter";
//					data.filter_plugin_name = child.attribute("type","").at(0);
//					toArgumentList(data.filter_args
//					process_childs = true;
//				}
//				else
//				if (node_type == "filter") {
//				}
//				else
//				if (node_type == "axis") {
//				}
		}
		default:
		{
		}
	}

	return newId;
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

