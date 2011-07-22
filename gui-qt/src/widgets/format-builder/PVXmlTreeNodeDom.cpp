///! \file PVXmlTreeNodeDom.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011
#include <PVXmlTreeNodeDom.h>

#define dbg {qDebug()<<__FILE__<<":"<<__LINE__;}




/******************************************************************************
 *
 * PVInspector::PVXmlTreeNodeDom::PVXmlTreeNodeDom
 *
 *****************************************************************************/
PVInspector::PVXmlTreeNodeDom::PVXmlTreeNodeDom() {
        setObjectName("PVXmlTreeNodeDom");
}




/******************************************************************************
 *
 * PVInspector::PVXmlTreeNodeDom::PVXmlTreeNodeDom
 *
 *****************************************************************************/
PVInspector::PVXmlTreeNodeDom::PVXmlTreeNodeDom(Type _type, const QString &_str,  QDomElement &dom, QDomDocument &xmlFile_) {
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
 * PVInspector::PVXmlTreeNodeDom::PVXmlTreeNodeDom
 *
 *****************************************************************************/
PVInspector::PVXmlTreeNodeDom::PVXmlTreeNodeDom(QDomElement const& dom) {
    setTypeFromString(dom.tagName());
    this->xmlDomElement = dom;
    parent = 0;
    isAlreadyExplored = false;
    isOnRoot=false;
}





/******************************************************************************
 *
 * PVInspector::PVXmlTreeNodeDom::~PVXmlTreeNodeDom
 *
 *****************************************************************************/
PVInspector::PVXmlTreeNodeDom::~PVXmlTreeNodeDom() {
    qDeleteAll(children);
}



/******************************************************************************
 *
 * PVInspector::PVXmlTreeNodeDom::addChild
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeNodeDom::addChild(PVInspector::PVXmlTreeNodeDom *child) {
    child->setParent(this);//set his parent
    child->setDoc(xmlFile);//set the dom node
    if (child->xmlDomElement.tagName() == "axis" || child->xmlDomElement.tagName() == "RegEx"|| child->xmlDomElement.tagName() == "url"|| child->xmlDomElement.tagName() =="field"){
      this->children.push_back(child);//child adding (in last position for axis, regex and url
    }
    else this->children.push_front(child);//child adding (in first for filter)
}


/******************************************************************************
 *
 * PVInspector::PVXmlTreeNodeDom::addChildAt
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeNodeDom::addChildAt(PVInspector::PVXmlTreeNodeDom *child, int row){
    this->children.insert(row,child);
}



/******************************************************************************
 *
 * PVInspector::PVXmlTreeNodeDom::getChild
 *
 *****************************************************************************/

PVInspector::PVXmlTreeNodeDom* PVInspector::PVXmlTreeNodeDom::getChild(int i){
    return this->getChildren().takeAt(i);
}



/******************************************************************************
 *
 * PVInspector::PVXmlTreeNodeDom::getChildren()
 *
 *****************************************************************************/
QList<PVInspector::PVXmlTreeNodeDom*> PVInspector::PVXmlTreeNodeDom::getChildren(){
    if(this->isAlreadyExplored) return this->children;
    else{explore();}//explore the tree structure.
    return this->children;
}



/******************************************************************************
 *
 * PVInspector::PVXmlTreeNodeDom::countChildren
 *
 *****************************************************************************/
int PVInspector::PVXmlTreeNodeDom::countChildren(){
    return this->getChildren().count();
}



/******************************************************************************
 *
 * PVInspector::PVXmlTreeNodeDom::getParent
 *
 *****************************************************************************/
PVInspector::PVXmlTreeNodeDom * PVInspector::PVXmlTreeNodeDom::getParent(){
    return this->parent;
}



/******************************************************************************
 *
 *  PVInspector::PVXmlTreeNodeDom::setParent
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeNodeDom::setParent(PVInspector::PVXmlTreeNodeDom *p_parent){
    this->parent = p_parent;
}



/******************************************************************************
 *
 * PVInspector::PVXmlTreeNodeDom::getName
 *
 *****************************************************************************/
QString PVInspector::PVXmlTreeNodeDom::getName() {
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
 * PVInspector::PVXmlTreeNodeDom::setName
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeNodeDom::setName(QString nom) {
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
 * PVInspector::PVXmlTreeNodeDom::explore
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeNodeDom::explore() {
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
 * PVInspector::PVXmlTreeNodeDom::setTypeFromString
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeNodeDom::setTypeFromString(QString nom){
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
 * PVInspector::PVXmlTreeNodeDom::deleteFromTree
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeNodeDom::deleteFromTree() {
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
 * PVInspector::PVXmlTreeNodeDom::removeChild
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeNodeDom::removeChild(PVXmlTreeNodeDom *child){
    this->children.removeOne(child);
}



/******************************************************************************
 *
 * PVInspector::PVXmlTreeNodeDom::getRow
 *
 *****************************************************************************/
int PVInspector::PVXmlTreeNodeDom::getRow(){
    return parent->children.indexOf(this,0);
}




/******************************************************************************
 *
 * PVInspector::PVXmlTreeNodeDom::newFilterAfter
 *
 *****************************************************************************/

void PVInspector::PVXmlTreeNodeDom::newFilterAfter(int indexeThisParmisFrere){
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
 * PVInspector::PVXmlTreeNodeDom::newSplitterAfter
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeNodeDom::newSplitterAfter(int indexeThisParmisFrere, QString tag){

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
 * PVInspector::PVXmlTreeNodeDom::setDoc
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeNodeDom::setDoc(QDomDocument &file){
    this->xmlFile = file;
}



/******************************************************************************
 *
 * PVInspector::PVXmlTreeNodeDom::getExpression
 *
 *****************************************************************************/
QString PVInspector::PVXmlTreeNodeDom::getExpression() {
    switch (this->type){
        case RegEx:;
        case filter:return this->xmlDomElement.attribute("expression", ".*");break;
        default:break;
    }
    return "";
}



/******************************************************************************
 *
 * PVInspector::PVXmlTreeNodeDom::setExpression
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeNodeDom::setExpression(QString exp) {
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
 * PVInspector::PVXmlTreeNodeDom::getNbr
 *
 *****************************************************************************/
int PVInspector::PVXmlTreeNodeDom::getNbr() {
    switch (this->type) {
        case RegEx:return this->xmlDomElement.childNodes().count();break;
        default:break;
    }
    return 0;
}

/******************************************************************************
 *
 * PVInspector::PVXmlTreeNodeDom::getOutName
 *
 *****************************************************************************/
QString PVInspector::PVXmlTreeNodeDom::getOutName() {
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
 * PVInspector::PVXmlTreeNodeDom::getOutWidget
 *
 *****************************************************************************/
PVInspector::PVXmlTreeNodeDom* PVInspector::PVXmlTreeNodeDom::getOutWidget(){
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
 * PVInspector::PVXmlTreeNodeDom::setNbr
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeNodeDom::setNbr(int nbr) {
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
 * PVInspector::PVXmlTreeNodeDom::getDom
 *
 *****************************************************************************/
QDomElement PVInspector::PVXmlTreeNodeDom::getDom(){
    return xmlDomElement;
}



/******************************************************************************
 *
 * PVInspector::PVXmlTreeNodeDom::addField
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeNodeDom::addField(int nbr) {
   
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
 * PVInspector::PVXmlTreeNodeDom::delField
 *
 *****************************************************************************/
/**
* delete the n last field.
* @param n
*/
void PVInspector::PVXmlTreeNodeDom::delField(int n) {
    for (int i = 0; i < n; i++) {
        PVXmlTreeNodeDom *lastChild=children.at(children.count()-1);
        removeChild(lastChild);
        getDom().removeChild(lastChild->getDom());
    }
}



/******************************************************************************
 *
 * PVInspector::PVXmlTreeNodeDom::addRegExRacine
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeNodeDom::addRegExRacine(){
    QDomElement elt = this->xmlFile.createElement("RegEx");//create the dom element
    PVXmlTreeNodeDom* child = new PVXmlTreeNodeDom(elt);//create the node 
    child->setParent(this);//place the node
    this->addChild(child);//...
}



/******************************************************************************
 *
 * PVInspector::PVXmlTreeNodeDom::addFilterRacine
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeNodeDom::addFilterRacine(){
    QDomElement elt = this->xmlFile.createElement("filter");//create the dom element
    PVXmlTreeNodeDom* child = new PVXmlTreeNodeDom(elt);//create the node
    child->isOnRoot=true;
    child->setParent(this);//place the node
    this->addChild(child);//...
}



/******************************************************************************
 *
 * PVInspector::PVXmlTreeNodeDom::typeToString
 *
 *****************************************************************************/
QString PVInspector::PVXmlTreeNodeDom::typeToString() {
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
 * PVInspector::PVXmlTreeNodeDom::setAttribute
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeNodeDom::setAttribute(QString name, QString value, bool flagSaveInXml) {
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
 * PVInspector::PVXmlTreeNodeDom::getAttribute
 *
 *****************************************************************************/
QString PVInspector::PVXmlTreeNodeDom::getAttribute(QString name, bool flagReadInXml) {
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
    
    return QString("");
}



/******************************************************************************
 *
 * PVInspector::PVXmlTreeNodeDom::isFieldOfUrl
 *
 *****************************************************************************/
bool PVInspector::PVXmlTreeNodeDom::isFieldOfUrl(){
    if(getParent()->xmlDomElement.tagName()=="url")return true;
    return false;
}