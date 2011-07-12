/* 
 * File:   PVCore::PVXmlParamParser.cpp
 * Author: rpernaudat
 * 
 * Created on 12 mai 2011, 11:27
 */

#include "pvcore/PVXmlParamParser.h"

PVCore::PVXmlParamParser::PVXmlParamParser(QString nameFile) {
    QFile fichier(nameFile);
    if(!fichier.exists()){
        //le fichier n'existe pas.
		return;
    }
    if (!fichier.open(QIODevice::ReadOnly | QIODevice::Text)) {
		return;
    }
    QTextStream tmpTextXml(&fichier);//creation of the file stream
    QDomDocument docXml;
    docXml.setContent(tmpTextXml.readAll());
    setDom(docXml.documentElement());

	fichier.close();
}

PVCore::PVXmlParamParser::~PVXmlParamParser() {
}
QHash<int, QStringList> const& PVCore::PVXmlParamParser::getTimeFormat()const{
    return time_format;
}
int PVCore::PVXmlParamParser::setDom(QDomElement node, int id) {
	//qDebug()<<"in "<<id<<"  "<<getNodeType(node)<<" "<<getNodeName(node);
	int newId = id;
	if (id == -1) {
		newId = 0;

		if (node.firstChild().toElement().tagName() == "field") {
			//qDebug() <<  " tag " << node.firstChild().toElement().tagName();
			newId = setDom(node.firstChild().toElement(), 0);
		} else {

			setDom(node, 0);
			//add splitter regx
//			PVCore::PVXmlParamParserData data;
//			data.id = newId;
//			data.exp = getNodeRegExp(node.firstChild().toElement());
//			data.type = PVCore::PVXmlParamParserData::splitter;
//			fields.push_back(data);
//			qDebug() << "add : " << newId << "  " << getNodeName(node.firstChild().toElement());
//
//			//recursive field
//			for (int i = 0; i < countChild(node.firstChild().toElement()); i++) {
//				//newId=setDom(node.firstChild().toElement().childNodes().at(i).toElement(),++newId);
//				newId = setDom(node.firstChild().toElement().childNodes().at(i).toElement(), newId);
//			}
		}



	}else{
		//add all filtersint defaultFormat;
		for(int i=0;i<countChild(node.toElement());i++){
			QDomElement child(node.childNodes().at(i).toElement());
			if(getNodeType(child)=="filter"){
				PVCore::PVXmlParamParserData data;
				data.id = newId;
				data.exp = getNodeRegExp(child.toElement());
				data.type = PVCore::PVXmlParamParserData::filter;
				data.grep_include = getNodeTypeGrep(child.toElement()).compare("include") == 0;
				fields.push_back(data);
				qDebug() << "add : " << newId << "  " << getNodeName(child.toElement());
			}
		}

		//add splitter regx (if there is)
		int defaultFormat = -1;
		for (int i = 0; i < countChild(node.toElement()); i++) {
			QDomElement child(node.childNodes().at(i).toElement());
			if (getNodeType(child) == "RegEx") {
				defaultFormat++;

				PVCore::PVXmlParamParserData data;
				data.id = newId;
				data.exp = getNodeRegExp(child.toElement());
				data.type = PVCore::PVXmlParamParserData::splitter;
				fields.push_back(data);
				qDebug() << "add : " << newId << "  " << getNodeName(child.toElement());
				//recursive field
				for (int iF = 0; iF < countChild(child.toElement()); iF++) {
					//qDebug() <<"name "<<getNodeName(child.childNodes().at(iF).toElement());
					//if(iF==0)newId--;
					//newId = setDom(child.childNodes().at(iF).toElement(), ++newId);
					newId = setDom(child.childNodes().at(iF).toElement(), newId);

				}
			}
		}

		//axes
		for (int i = 0; i < countChild(node.toElement()); i++) {
			QDomElement child(node.childNodes().at(i).toElement());
			if (getNodeType(child) == "axis") {
				defaultFormat++;

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
				// qDebug()<<"debut";
				if(child.attribute("type","")=="time"){
					PVLOG_DEBUG("Time format for axis %d\n", newId);
					time_format[newId+1]  = child.attribute("time-format","").split("\n");
				}

				newId++;
			}
		}

		for (int i = 0; i < countChild(node.toElement()); i++) {
			QDomElement child(node.childNodes().at(i).toElement());
			if (getNodeType(child) == "url") {
				defaultFormat++;
				PVCore::PVXmlParamParserData data;
				data.id = newId;
				data.exp = "";
				data.type = PVCore::PVXmlParamParserData::splitter_url;
				fields.push_back(data);
				qDebug() << "add : " << newId << "  url ";
				PVLOG_WARN("FIXME: fixed size value for URL splitter\n");
				for (int iF = 0; iF < PVFORMAT_NUMBER_FIELD_URL; iF++) {
					newId = setDom(child.childNodes().at(iF).toElement(), newId);
				}

			}
		}

		for (int i = 0; i < countChild(node.toElement()); i++) {
			QDomElement child(node.childNodes().at(i).toElement());
			if (getNodeType(child) == "pcap") {
				defaultFormat++;
				PVCore::PVXmlParamParserData data;
				data.id = newId;
				data.exp = "";
				data.type = PVCore::PVXmlParamParserData::splitter_pcap;
				fields.push_back(data);
				qDebug() << "add : " << newId << "  pcap ";
				//process each field
				for (int iF = 0; iF < countChild(child.toElement()); iF++) {
					newId = setDom(child.childNodes().at(iF).toElement(), newId);
				}
			}
		}

		for (int i = 0; i < countChild(node.toElement()); i++) {
			QDomElement child(node.childNodes().at(i).toElement());
			if (getNodeType(child) == "csv") {
				defaultFormat++;
				PVCore::PVXmlParamParserData data;
				data.id = newId;
				data.csv_delimiter = child.attribute("delimiter","").at(0);
				data.type = PVCore::PVXmlParamParserData::splitter_csv;
				fields.push_back(data);
				qDebug() << "add : " << newId << "  csv ";
				//process each field
				for (int iF = 0; iF < countChild(child.toElement()); iF++) {
					newId = setDom(child.childNodes().at(iF).toElement(), newId);
				}
			}
		}

		if (defaultFormat >1 ) {
			//qDebug()<<"default : to much axis or RegEx on a field.  " << QString(defaultFormat);
		}
	}

    return newId;
}

QString PVCore::PVXmlParamParser::toString(){
    QString s;
    for (int i = 0; i < fields.count(); i++) {
        s.push_back(fields.at(i).id);
        s.push_back("   ");
        s.push_back(fields.at(i).exp);
        s.push_back("   ");
        s.push_back(fields.at(i).type);
    }
    return s;
}

QList<QHash<QString, QString> > const& PVCore::PVXmlParamParser::getAxes()const{
    return this->axes;
}


/******************************* private *************************************/
int PVCore::PVXmlParamParser::countChild(QDomElement node){
    //if(getNodeType(node)=="RegEx"){
        return node.childNodes().count();
    //}
    return 0;
}

QList<PVCore::PVXmlParamParserData> const& PVCore::PVXmlParamParser::getFields() const {
    return fields;
}
QString PVCore::PVXmlParamParser::getNodeRegExp(QDomElement node){
    return node.attribute("expression","");
}
QString PVCore::PVXmlParamParser::getNodeTypeGrep(QDomElement node){
    return node.attribute("type","");
}
QString PVCore::PVXmlParamParser::getNodeName(QDomElement node){
    return node.attribute("name","");
}
QString PVCore::PVXmlParamParser::getNodeType(QDomElement node){
    return node.tagName();
}

