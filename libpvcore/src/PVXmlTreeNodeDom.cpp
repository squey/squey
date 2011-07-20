///! \file PVXmlTreeNodeDom.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011
#include <pvcore/PVXmlTreeNodeDom.h>

#define dbg {qDebug()<<__FILE__<<":"<<__LINE__;}




/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::PVXmlTreeNodeDom
 *
 *****************************************************************************/
PVCore::PVXmlTreeNodeDom::PVXmlTreeNodeDom() {
        setObjectName("PVXmlTreeNodeDom");
}




/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::PVXmlTreeNodeDom
 *
 *****************************************************************************/
PVCore::PVXmlTreeNodeDom::PVXmlTreeNodeDom(Type _type, const QString &_str,  QDomElement &dom, QDomDocument &xmlFile_) {
    this->type = _type;
    this->str=_str;
    this->xmlDomElement=dom;
    this->xmlFile=xmlFile_;
    parent=0;
    isAlreadyExplored=false;
    isOnRoot=false;
}



/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::PVXmlTreeNodeDom
 *
 *****************************************************************************/
PVCore::PVXmlTreeNodeDom::PVXmlTreeNodeDom(QDomElement const& dom) {
    setTypeFromString(dom.tagName());
    this->xmlDomElement = dom;
    parent = 0;
    isAlreadyExplored = false;
    isOnRoot=false;
}





/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::~PVXmlTreeNodeDom
 *
 *****************************************************************************/
PVCore::PVXmlTreeNodeDom::~PVXmlTreeNodeDom() {
    qDeleteAll(children);
}



/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::addChild
 *
 *****************************************************************************/
void PVCore::PVXmlTreeNodeDom::addChild(PVCore::PVXmlTreeNodeDom *child) {
    child->setParent(this);//set his parent
    child->setDoc(xmlFile);//set the dom node
    if (child->xmlDomElement.tagName() == "axis" || child->xmlDomElement.tagName() == "RegEx"|| child->xmlDomElement.tagName() == "url"|| child->xmlDomElement.tagName() =="field"){
      this->children.push_back(child);//child adding (in last position for axis, regex and url
    }
    else this->children.push_front(child);//child adding (in first for filter)
}


/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::addChildAt
 *
 *****************************************************************************/
void PVCore::PVXmlTreeNodeDom::addChildAt(PVCore::PVXmlTreeNodeDom *child, int row){
    this->children.insert(row,child);
}



/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::getChild
 *
 *****************************************************************************/

PVCore::PVXmlTreeNodeDom* PVCore::PVXmlTreeNodeDom::getChild(int i){
    return this->getChildren().takeAt(i);
}



/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::getChildren()
 *
 *****************************************************************************/
QList<PVCore::PVXmlTreeNodeDom*> PVCore::PVXmlTreeNodeDom::getChildren(){
    if(this->isAlreadyExplored) return this->children;
    else{explore();}//explore the tree structure.
    return this->children;
}



/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::countChildren
 *
 *****************************************************************************/
int PVCore::PVXmlTreeNodeDom::countChildren(){
    return this->getChildren().count();
}



/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::getParent
 *
 *****************************************************************************/
PVCore::PVXmlTreeNodeDom * PVCore::PVXmlTreeNodeDom::getParent(){
    return this->parent;
}



/******************************************************************************
 *
 *  PVCore::PVXmlTreeNodeDom::setParent
 *
 *****************************************************************************/
void PVCore::PVXmlTreeNodeDom::setParent(PVCore::PVXmlTreeNodeDom *p_parent){
    this->parent = p_parent;
}



/******************************************************************************
 *
 * PVInspector::PVXmlTreeNodeDom::getName
 *
 *****************************************************************************/
QString PVCore::PVXmlTreeNodeDom::getName() {
    //if(this->type==Root)return "root";
    switch (this->type){
        case RegEx:;
        case axis:;
        case url:;
        case filter:return this->xmlDomElement.attribute("name", "");break;//get the attribute name in the DOM
        case field:
            if(isFieldOfUrl()){//if node is an url...
                switch(getRow()){//...return name as following
                    case 0:return "protocol";
                    case 1:return "domain";
                    case 2:return "tld";
                    case 3:return "port";
                    case 4:return "url";
                    case 5:return "variable";
                }
                return "url field";
            }
            else {//... else get the attribute "name".
	      QString selectionRegExpName = QString("selectionRegExp-%0").arg(getRow()+1);
	      QString selectionRegExp = getParent()->getAttribute(selectionRegExpName, "");
	      return selectionRegExp;
	    }
        default:return this->str;
    }
    return this->str;
}



/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::setName
 *
 *****************************************************************************/
void PVCore::PVXmlTreeNodeDom::setName(QString nom) {
    switch (this->type){
        case RegEx:;// // set the attribute name
        case axis:;// // set the attribute name
        case filter:this->xmlDomElement.setAttribute("name",nom);break;// set the attribute name
        case field: ;
        default:this->str = nom;
    }
}



/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::explore
 *
 *****************************************************************************/
void PVCore::PVXmlTreeNodeDom::explore() {
    this->isAlreadyExplored = true;

    /* search for child in the dom.*/
    QDomElement childItem = this->xmlDomElement.firstChildElement();
    while (!childItem.isNull()) {
        addChild(new PVXmlTreeNodeDom(childItem));
        childItem = childItem.nextSiblingElement();
    }
}




/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::setTypeFromString
 *
 *****************************************************************************/
void PVCore::PVXmlTreeNodeDom::setTypeFromString(QString nom){
    if(nom==QString("RegEx"))type = RegEx;
    else if(nom==QString("filter"))type = filter;
    else if(nom==QString("field"))type = field;
    else if(nom==QString("axis"))type = axis;
    else if(nom==QString("url"))type = url;
    
    
    //else if(nom==QString("root"))type = Root;
    else std::cerr<<"le type '"<<nom.toStdString()<<"' n'existe pas."<<std::endl;
}



/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::deleteFromTree
 *
 *****************************************************************************/
void PVCore::PVXmlTreeNodeDom::deleteFromTree() {
    if (this->getParent()->getDom().tagName() != "param") {
        this->xmlDomElement.parentNode().removeChild(this->xmlDomElement);
        this->getParent()->removeChild(this);
    }else{
        this->getParent()->getDom().removeChild(this->xmlDomElement);
        this->getParent()->removeChild(this);
    }
}



/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::removeChild
 *
 *****************************************************************************/
void PVCore::PVXmlTreeNodeDom::removeChild(PVXmlTreeNodeDom *child){
    this->children.removeOne(child);
}



/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::getRow
 *
 *****************************************************************************/
int PVCore::PVXmlTreeNodeDom::getRow(){
    return parent->children.indexOf(this,0);
}




/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::newFilterAfter
 *
 *****************************************************************************/

void PVCore::PVXmlTreeNodeDom::newFilterAfter(int indexeThisParmisFrere){
    //create the new dom element
    QDomElement domEltNewFilter=this->xmlFile.createElement("filter");//create the dom
    domEltNewFilter.setAttribute("name","new filter");//set the name
    this->xmlDomElement.parentNode().insertAfter(domEltNewFilter,this->xmlDomElement);//push new dom in dom tree 
    
    //add the new 
    PVXmlTreeNodeDom *newNodeFilter = new PVXmlTreeNodeDom(domEltNewFilter);//create the node
    newNodeFilter->setParent(this->parent);//set the parent
    this->parent->children.insert(indexeThisParmisFrere+1,newNodeFilter);//push node in tree
}




/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::newSplitterAfter
 *
 *****************************************************************************/
void PVCore::PVXmlTreeNodeDom::newSplitterAfter(int indexeThisParmisFrere, QString tag){

    //create the new dom element
    QDomElement domEltNewSplit=this->xmlFile.createElement(tag);//create the dom
    domEltNewSplit.setAttribute("name","new Splitter");//set the name
    this->xmlDomElement.parentNode().insertAfter(domEltNewSplit,this->xmlDomElement);//push new dom in dom tree 
    
    //add the new 
    PVXmlTreeNodeDom *newNodeSplitter = new PVXmlTreeNodeDom(domEltNewSplit);//create the node
    newNodeSplitter->setParent(this->parent);//set the parent
    this->parent->children.insert(indexeThisParmisFrere+1,newNodeSplitter);//push node in tree
}




/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::setDoc
 *
 *****************************************************************************/
void PVCore::PVXmlTreeNodeDom::setDoc(QDomDocument &file){
    this->xmlFile = file;
}



/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::getExpression
 *
 *****************************************************************************/
QString PVCore::PVXmlTreeNodeDom::getExpression() {
    switch (this->type){
        case RegEx:;
        case filter:return this->xmlDomElement.attribute("expression", ".*");break;
        default:break;
    }
    return "";
}



/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::setExpression
 *
 *****************************************************************************/
void PVCore::PVXmlTreeNodeDom::setExpression(QString exp) {
    switch (this->type) {
        case RegEx:this->xmlDomElement.setAttribute("expression", exp);
            break;
        case filter:this->xmlDomElement.setAttribute("expression", exp);
            break;
        case field: break;
        default:break;
    }
}


/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::getNbr
 *
 *****************************************************************************/
int PVCore::PVXmlTreeNodeDom::getNbr() {
    switch (this->type) {
        case RegEx:return this->xmlDomElement.childNodes().count();break;
        default:break;
    }
    return 0;
}

/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::getOutName
 *
 *****************************************************************************/
QString PVCore::PVXmlTreeNodeDom::getOutName() {
    QString l;
    if (this->type == field) {// its a field whiche is selected...
        for(int i=0;i<children.count();i++){
            if(children.at(i)->getDom().tagName()=="axis")return children.at(i)->getDom().attribute("name"," ");//return the attribute name.
            if(children.at(i)->getDom().tagName()=="url")return "URL";
            if(children.at(i)->getDom().tagName()=="RegEx")return children.at(i)->getDom().attribute("name"," ");
        }
    }
    return l;
}

/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::getOutWidget
 *
 *****************************************************************************/
PVCore::PVXmlTreeNodeDom* PVCore::PVXmlTreeNodeDom::getOutWidget(){
  PVXmlTreeNodeDom *l;
    if (this->type == field) {// its a field whiche is selected...
        for(int i=0;i<children.count();i++){
            if((children.at(i)->getDom().tagName()=="axis")||//if it's an axis
            (children.at(i)->getDom().tagName()=="url")||//if it's an url
            (children.at(i)->getDom().tagName()=="RegEx")){//if it's a regexp
	      return children.at(i);//return the node
	    }
        }
    }
    return l;
}



/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::setNbr
 *
 *****************************************************************************/
void PVCore::PVXmlTreeNodeDom::setNbr(int nbr) {
  //this function update the number of field in the regexp
    switch (this->type) {
        case RegEx:
            if (nbr > this->getNbr()) {
                addField(nbr - this->getNbr());//add some fields
            }else if(nbr < this->getNbr()){
                delField(this->getNbr()-nbr);//delete some fiels
            }
            break;
        default:break;
    }
}




/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::getDom
 *
 *****************************************************************************/
QDomElement PVCore::PVXmlTreeNodeDom::getDom(){
    return xmlDomElement;
}



/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::addField
 *
 *****************************************************************************/
void PVCore::PVXmlTreeNodeDom::addField(int nbr) {
   
    if (this->type == RegEx && nbr > 0) {
        for (int i = 0; i < nbr; i++) {
            //dom field
            QDomElement newField = this->xmlFile.createElement("field");
            xmlDomElement.appendChild(newField); //placement dans le Dom
            //dom axis
            QDomElement newAxisDom = this->xmlFile.createElement("axis");
            newField.appendChild(newAxisDom); //placement dans le Dom
            
            
            
            //node field
            PVXmlTreeNodeDom *newNodeField = new PVXmlTreeNodeDom(newField); //création
            newNodeField->setParent(this); //modif du parent
            //node axis (we add an axis in each field)
            PVXmlTreeNodeDom *newAxis = new PVXmlTreeNodeDom(newAxisDom);
            newAxis->setParent(newNodeField);
            newAxis->setName("");

            this->children.push_back(newNodeField); //placement dans l'arbre du View
            //newNodeField->children.push_back(newAxis);
        }
    }
}



/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::delField
 *
 *****************************************************************************/
/**
* delete the n last field.
* @param n
*/
void PVCore::PVXmlTreeNodeDom::delField(int n) {
    for (int i = 0; i < n; i++) {
        PVXmlTreeNodeDom *lastChild=children.at(children.count()-1);
        removeChild(lastChild);
        getDom().removeChild(lastChild->getDom());
    }
}



/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::addRegExRacine
 *
 *****************************************************************************/
void PVCore::PVXmlTreeNodeDom::addRegExRacine(){
    QDomElement elt = this->xmlFile.createElement("RegEx");//create the dom element
    PVXmlTreeNodeDom* child = new PVXmlTreeNodeDom(elt);//create the node 
    child->setParent(this);//place the node
    this->addChild(child);//...
}



/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::addFilterRacine
 *
 *****************************************************************************/
void PVCore::PVXmlTreeNodeDom::addFilterRacine(){
    QDomElement elt = this->xmlFile.createElement("filter");//create the dom element
    PVXmlTreeNodeDom* child = new PVXmlTreeNodeDom(elt);//create the node
    child->isOnRoot=true;
    child->setParent(this);//place the node
    this->addChild(child);//...
}



/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::typeToString
 *
 *****************************************************************************/
QString PVCore::PVXmlTreeNodeDom::typeToString() {
  //return the type as a QString
    if (type == RegEx)return "RegEx";
    if (type == filter)return "filter";
    if (type == field)return "field";
    if (type == axis)return "axis";
    if (type == Root)return "Root";
    if(type == url)return "url";
    return "";
}



/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::setAttribute
 *
 *****************************************************************************/
void PVCore::PVXmlTreeNodeDom::setAttribute(QString name, QString value, bool flagSaveInXml) {
  //PVLOG_INFO("PVXmlTreeNodeDom::setAttribute('%s','%s')\n",name.toUtf8().constData(),value.toUtf8().constData());
      if(flagSaveInXml){
	  xmlDomElement.setAttribute(name, value);
      }else{
	  xmlDomElement.setAttribute(name, QString(""));
      }
      otherData.insert(name,value);
}



/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::getAttribute
 *
 *****************************************************************************/
QString PVCore::PVXmlTreeNodeDom::getAttribute(QString name, bool flagReadInXml) {
  //PVLOG_INFO("PVXmlTreeNodeDom::getAttribute( %s )\n",name.toUtf8().constData());
    if(flagReadInXml){
	//update the savedvalue
        if(otherData.contains(name)){
	    otherData[name] = xmlDomElement.attribute(name,"");
	}else{
	    otherData.insert(name,xmlDomElement.attribute(name,""));
	}
	//read the dom element
	return xmlDomElement.attribute(name,"");
    }else  if(otherData.contains(name)){
	return otherData[name];
    }
    
    return QString();
}



/******************************************************************************
 *
 * PVCore::PVXmlTreeNodeDom::isFieldOfUrl
 *
 *****************************************************************************/
bool PVCore::PVXmlTreeNodeDom::isFieldOfUrl(){
    if(getParent()->xmlDomElement.tagName()=="url")return true;
    return false;
}

void PVCore::PVXmlTreeNodeDom::setFromArgumentList(PVArgumentList const& args)
{
	PVArgumentList::const_iterator it;
	for (it = args.begin(); it != args.end(); it++) {
		setAttribute(it.key(), PVCore::PVArgument_to_QString(it.value()), true);
	}
}

void PVCore::PVXmlTreeNodeDom::toArgumentList(PVArgumentList const& default_args, PVArgumentList& args)
{
	PVArgumentList::const_iterator it;
	for (it = default_args.begin(); it != default_args.end(); it++) {
		QString const& key = it.key();
		QString v = getAttribute(key, true);
		QVariant vset;
		if (v.isNull()) {
			vset = it.value();
		}
		else {
			vset.setValue(v);
		}
		args[key] = vset;
	}
}
