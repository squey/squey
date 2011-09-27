//! \file PVXmlDomModel.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011
#include <PVXmlDomModel.h>
#include <pvkernel/filter/PVFieldsFilterParamWidget.h>
#include <pvkernel/rush/PVFormatVersion.h>
#include <pvkernel/rush/PVXmlTreeNodeDom.h>

#include <QString>

#define dbg()  {qDebug()<<__FILE__<<__LINE__;}


/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::PVXmlDomModel
 *
 *****************************************************************************/
PVInspector::PVXmlDomModel::PVXmlDomModel(QWidget * parent): QAbstractItemModel(parent) {

	PVRush::PVXmlTreeNodeDom* m_rootNode = PVRush::PVXmlTreeNodeDom::new_format(xmlFile);
	xmlRootDom = xmlFile.documentElement();
    setRoot(m_rootNode);
    setObjectName("PVXmlDomModel");
}


/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::~PVXmlDomModel
 *
 *****************************************************************************/
PVInspector::PVXmlDomModel::~PVXmlDomModel() {
}







/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::index
 *
 *****************************************************************************/
/**
* create a new index for the treeview.
*/
QModelIndex PVInspector::PVXmlDomModel::index(int r, int c, const QModelIndex& parent) const{

    if(!rootNode||r<0||c<0)return QModelIndex();

    PVRush::PVXmlTreeNodeDom *parentNode = nodeFromIndex(parent);
    PVRush::PVXmlTreeNodeDom *childNode = parentNode->getChildren().value(r);

    if(!childNode)return QModelIndex();



    return createIndex(r,c, childNode);
}



/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::parent
 *
 *****************************************************************************/
/**
* Get the parent index
* @param child node
*/
QModelIndex PVInspector::PVXmlDomModel::parent(const QModelIndex& child)const {

    PVRush::PVXmlTreeNodeDom *node=nodeFromIndex(child);
    if(!node)return QModelIndex();

    PVRush::PVXmlTreeNodeDom *parentNode = node->getParent();
    if(!parentNode)return QModelIndex();

    PVRush::PVXmlTreeNodeDom *grandParentNode = parentNode->getParent();
    if(!grandParentNode)return QModelIndex();

    int row = grandParentNode->getChildren().indexOf(parentNode);
    return createIndex(row,0,parentNode);
}




/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::rowCount
 *
 *****************************************************************************/

int PVInspector::PVXmlDomModel::rowCount(const QModelIndex& parent)const {

    if( parent.column()>0)return 0;

    PVRush::PVXmlTreeNodeDom *parentNode=nodeFromIndex(parent);
    if(!parentNode)return 0;

    return parentNode->countChildren();
}


/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::columnCount
 *
 *****************************************************************************/
/**
* Return the number of raw.
*/
int PVInspector::PVXmlDomModel::columnCount(const QModelIndex& )const {
    return 2;
}


/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::countParent
 *
 *****************************************************************************/
/**
* return the number of node from the root to the child.
*/
int PVInspector::PVXmlDomModel::countParent(const QModelIndex &index) {
    if (index.parent().isValid()&& nodeFromIndex(index)->getName()!="root" ) {
        return (countParent(index.parent()) + 1);
    }
    return 0;
}


/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::data
 *
 *****************************************************************************/
QVariant PVInspector::PVXmlDomModel::data(const QModelIndex &index, int role)const {

    if (index.isValid()) {
        if (role == Qt::DisplayRole || role == Qt::EditRole) {
            PVRush::PVXmlTreeNodeDom *node = nodeFromIndex(index);
            assert(node);
            if (index.column() == 0) {
                if (node->typeToString() == "RegEx") {
                    return "Splitter (RegEx)";
                } else if (node->typeToString() == "url") {
                    return "Splitter (URL)";
                } else if (node->typeToString() == "splitter") {
                    QString s = QString("splitter : %1").arg(node->attribute("type", ""));
                    return s;
                }
                return node->typeToString();
               
            } else if(index.column() == 1){
                if (node->typeToString() == "field") {
                    return QString("");
                }
		//if(node->typeToString()=="RegEx")return node->getAttribute(QString("expression"));
                return node->getName();
            }else {
                return QString("");
            }
        }
    } else return QVariant();
    return QVariant();
}




/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::setRoot
 *
 *****************************************************************************/
/**
* Setup the root.
*/
void PVInspector::PVXmlDomModel::setRoot(PVRush::PVXmlTreeNodeDom *node){
    this->rootNode = node;
    reset();
}


/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::nodeFromIndex
 *
 *****************************************************************************/
/**
* Return the node from the index.
* @param index
*/
PVRush::PVXmlTreeNodeDom* PVInspector::PVXmlDomModel::nodeFromIndex(const QModelIndex &index)const
{
    if(index.isValid()){
        assert(index.internalPointer());
        //PVLOG_DEBUG("                    nodeFromIndex            %s        \n",qPrintable(static_cast<PVRush::PVXmlTreeNodeDom *>(index.internalPointer())->getName()));
        //PVLOG_DEBUG("              ---      \n");
        return static_cast<PVRush::PVXmlTreeNodeDom *>(index.internalPointer());
    }else{
        return rootNode;
    }
}




/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::flags
 *
 *****************************************************************************/
Qt::ItemFlags PVInspector::PVXmlDomModel::flags ( const QModelIndex & index ) const{
    Qt::ItemFlags flags = Qt::ItemIsSelectable|Qt::ItemIsEnabled;
    if(index.column()==1){
            if(nodeFromIndex(index)->isEditable()){
                   flags = flags|Qt::ItemIsEditable; 
            }
    }
    return flags;
}



/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::headerData
 *
 *****************************************************************************/
QVariant PVInspector::PVXmlDomModel::headerData ( int section, Qt::Orientation orientation, int role ) const{
        //If it's view is displaying
        if(role == Qt::DisplayRole){
                //if the orientation is horizontal
                if(orientation == Qt::Horizontal){
                        switch(section){
                                case 0://edit first column
                                        return QVariant("Type");
                                        break;
                                case 1://edit second column
                                        return QVariant("Name");
                                        break;
                        }
                }
        }
        return QVariant();
}


/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::supportedDropActions
 *
 *****************************************************************************/
Qt::DropActions PVInspector::PVXmlDomModel::supportedDropActions() const{
    return Qt::CopyAction | Qt::MoveAction;
}



/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::setData
 *
 *****************************************************************************/
bool PVInspector::PVXmlDomModel::setData(const QModelIndex & index, const QVariant & value, int role){
    if(role==Qt::EditRole){
        if(index.column()==1){//just choose the second column
            nodeFromIndex(index)->setName(value.toString());
            emit dataChanged(index ,index );
            return true;
        }
    }
    return false;
}



/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::saveXml
 *
 *****************************************************************************/
bool PVInspector::PVXmlDomModel::saveXml(QString nomDuFichierXml){
    QFile file(nomDuFichierXml);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
		QMessageBox msg(QMessageBox::Critical, "Error while saving format", QString("Unable to open %1 for writing: ").arg(nomDuFichierXml) + file.errorString(), QMessageBox::Ok);
		msg.exec();
		return false;
	}
    QString version = QString("%1").arg(PVFORMAT_CURRENT_VERSION);
    xmlRootDom.setAttribute("version",version);
	QByteArray data(xmlFile.toString().toUtf8());
	int size_written = file.write(data);
	int data_size = data.size();
    if (size_written != data_size || !file.flush()) {
		QMessageBox msg(QMessageBox::Critical, "Error while saving format", QString("An error occured while saving format: ") + file.errorString(), QMessageBox::Ok);
		msg.exec();
		return false;
	}
    file.close();

	return true;
}



/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::applyModification
 *
 *****************************************************************************/
void PVInspector::PVXmlDomModel::applyModification(QModelIndex &,PVXmlParamWidget *){
    emit layoutChanged();
}



/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::saveDefault
 *
 *****************************************************************************/
void PVInspector::PVXmlDomModel::saveDefault(){
    std::cout<<"saveDefault()"<<std::endl;
    saveXml("tmp.xml");
}


/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::deleteSelectio
 *
 *****************************************************************************/
void PVInspector::PVXmlDomModel::deleteSelection(QModelIndex &index) {
    if (index.isValid()) {
        PVRush::PVXmlTreeNodeDom *nodeASupprimer = nodeFromIndex(index);
        if (nodeASupprimer != rootNode) {
            nodeASupprimer->deleteFromTree();
            emit layoutChanged();
        }
    }else{
    
    }
}


/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::addAxisIn
 *
 *****************************************************************************/
void PVInspector::PVXmlDomModel::addAxisIn(const QModelIndex &index) {
	if (index.isValid()) {//if index valid, add axis in field... 
		PVRush::PVXmlTreeNodeDom *field = nodeFromIndex(index);

		//make sure that there not already axis or regexp.
		if (!trustConfictSplitAxes(index))return;

		//axis adding
		if (field->typeToString() != "field") {
			message("You must select a field first.");
			return;
		}

		addAxisIn(field);

	} else {//else add on root node...
		if (!trustConfictSplitAxes(index))return;
		addAxisIn(NULL);
	}
	emit layoutChanged();
}


/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::addAxisIn
 *
 *****************************************************************************/
PVRush::PVXmlTreeNodeDom* PVInspector::PVXmlDomModel::addAxisIn(PVRush::PVXmlTreeNodeDom* parentNode)
{
	PVRush::PVXmlTreeNodeDom* child;
	QDomElement newAxis = xmlFile.createElement("axis");
	child = new PVRush::PVXmlTreeNodeDom(newAxis);
	if (parentNode != NULL) {
		assert(parentNode->typeToString() == "field");
		child->isOnRoot = false;
	}
	else {
		child->isOnRoot=true;
		parentNode = rootNode;
	}
	child->setParent(parentNode);
	parentNode->addChild(child);
	parentNode->getDom().appendChild(newAxis);

	return child;
}


/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::addFilterAfter
 *
 *****************************************************************************/ 
void PVInspector::PVXmlDomModel::addFilterAfter(QModelIndex &index) {
    PVRush::PVXmlTreeNodeDom *childPrecedent = nodeFromIndex(index); //node sélectionné
    if(childPrecedent->typeToString()!="field"){
        message("You must select a field first.")
    }else if(childPrecedent->typeToString()=="field"){
        //dom
        QDomElement newDom = xmlFile.createElement("filter");
        newDom.setAttribute(QString("type"),QString("regexp"));
        childPrecedent->getDom().appendChild(newDom);
        
        //tree
        PVRush::PVXmlTreeNodeDom* child = new PVRush::PVXmlTreeNodeDom(newDom);
        child->setParent(childPrecedent);
        childPrecedent->addChild(child);
    }
        
        
    emit layoutChanged(); //refresh
}


/******************************************************************************
 *
 *  PVInspector::PVXmlDomModel::addSplitter
 *
 *****************************************************************************/
PVRush::PVXmlTreeNodeDom* PVInspector::PVXmlDomModel::addSplitter(const QModelIndex &index, PVFilter::PVFieldsSplitterParamWidget_p splitterPlugin)
{
	assert(splitterPlugin);

	PVRush::PVXmlTreeNodeDom* child;
	PVLOG_DEBUG("PVInspector::PVXmlDomModel::addSplitter\n");
	PVRush::PVXmlTreeNodeDom *field;
	if(index.isValid()){//add as child
		field = nodeFromIndex(index);
		if (field->typeToString() == "field") {//a splitter can be add only in field...
			if (!trustConfictSplitAxes(index)){
				//message("You must select a field with axis or splitter inside.");
				return NULL;//we can't add more than one splitter in a field
			}
			PVLOG_DEBUG("     adding splitter in a field\n");
		} else {
			//message(QString(field->getDom().tagName()));
			message("You must select a field first.");
			return NULL;
		}
	}else{//add on the root

		if (!trustConfictSplitAxes(index))return NULL;//we can't add more than one splitter in a field
		field = rootNode;

	}
	PVLOG_DEBUG("     adding splitter on root node\n");
	//add node in dom
	QDomElement newDom = xmlFile.createElement(splitterPlugin->type_name());
	QString registered_name = splitterPlugin->registered_name();
	PVLOG_DEBUG("          set tag %s, type %s\n", qPrintable(splitterPlugin->type_name()), qPrintable(registered_name));
	newDom.setAttribute("type", registered_name);
	field->getDom().appendChild(newDom);
	//
	//add node in tree
	child = new PVRush::PVXmlTreeNodeDom(newDom);
	child->setParent(field);
	field->addChild(child);

	//save the splitter plugin referance
	child->setSplitterPlugin(splitterPlugin);

	emit layoutChanged();
	return child;
}


PVRush::PVXmlTreeNodeDom* PVInspector::PVXmlDomModel::addSplitterWithAxes(const QModelIndex& index, PVFilter::PVFieldsSplitterParamWidget_p splitterPlugin, QStringList axesName)
{
	PVRush::PVXmlTreeNodeDom* splitter_node = addSplitter(index, splitterPlugin);

	for (int i = 0; i < axesName.size(); i++) {
		// TODO: we should be able to create a field and/or an axis from separate functions !!!!!
		QDomElement newField = xmlFile.createElement(PVFORMAT_XML_TAG_FIELD_STR);
		QDomElement newAxis = xmlFile.createElement(PVFORMAT_XML_TAG_AXIS_STR);
		newAxis.setAttribute(PVFORMAT_AXIS_NAME_STR, axesName[i]);
		setDefaultAttributesForAxis(newAxis);

		newField.appendChild(newAxis);
		splitter_node->getDom().appendChild(newField);
	}
	return splitter_node;
}

void PVInspector::PVXmlDomModel::setDefaultAttributesForAxis(QDomElement& elt)
{
	assert(elt.tagName() == PVFORMAT_XML_TAG_AXIS_STR);
	elt.setAttribute(PVFORMAT_AXIS_TITLECOLOR_STR, PVFORMAT_AXIS_TITLECOLOR_DEFAULT);
	elt.setAttribute(PVFORMAT_AXIS_COLOR_STR, PVFORMAT_AXIS_COLOR_DEFAULT);
	elt.setAttribute(PVFORMAT_AXIS_KEY_STR, PVFORMAT_AXIS_KEY_DEFAULT);
	elt.setAttribute(PVFORMAT_AXIS_MAPPING_STR, PVFORMAT_AXIS_MAPPING_DEFAULT);
	elt.setAttribute(PVFORMAT_AXIS_PLOTTING_STR, PVFORMAT_AXIS_PLOTTING_DEFAULT);
	elt.setAttribute(PVFORMAT_AXIS_TYPE_STR, PVFORMAT_AXIS_TYPE_DEFAULT);
	elt.setAttribute(PVFORMAT_AXIS_GROUP_STR, PVFORMAT_AXIS_GROUP_DEFAULT);
}


/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::addRegExIn
 *
 *****************************************************************************/
void PVInspector::PVXmlDomModel::addRegExIn(const QModelIndex &index) {
    //if it's a "field"
    if (index.isValid()) {
        PVRush::PVXmlTreeNodeDom *field = nodeFromIndex(index);
        if (field->typeToString() == "field") {
            if (!trustConfictSplitAxes(index))return;
            //dom
            QDomElement newDom = xmlFile.createElement("splitter");
			newDom.setAttribute("type", "regexp");
            field->getDom().appendChild(newDom);

            //tree
            PVRush::PVXmlTreeNodeDom* child = new PVRush::PVXmlTreeNodeDom(newDom);

            child->setParent(field);
            field->addChild(child);
        } else {
            //message(QString(field->getDom().tagName()));
            message("You must select a field first.");
        }
    } else {
        if (!trustConfictSplitAxes(index))return;
        //rootNode->addRegExRacine();
        //dom
		QDomElement newDom = xmlFile.createElement("splitter");
		newDom.setAttribute("type", "regexp");
        rootNode->getDom().appendChild(newDom);
//
//        //tree
        PVRush::PVXmlTreeNodeDom* child = new PVRush::PVXmlTreeNodeDom(newDom);
        child->setParent(rootNode);
        rootNode->addChild(child);
    }
    emit layoutChanged();
}




/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::moveDown
 *
 *****************************************************************************/
void PVInspector::PVXmlDomModel::moveDown(const QModelIndex &index) {
    //qDebug() << "model MoveDown";

    PVRush::PVXmlTreeNodeDom *child = nodeFromIndex(index);
    PVRush::PVXmlTreeNodeDom *parent = child->getParent();
    PVRush::PVXmlTreeNodeDom *fllower;
    if (child->getRow() + 1 < parent->getChildren().count()) {
        fllower = parent->getChild(index.row() + 1);
        //dom effect
        parent->getDom().removeChild(child->getDom());
        parent->getDom().insertAfter(child->getDom(), fllower->getDom());
        //node effect
        parent->removeChild(child);
        parent->addChildAt(child, fllower->getRow() + 1);
    }
    emit layoutChanged();
}


/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::moveUp
 *
 *****************************************************************************/
void PVInspector::PVXmlDomModel::moveUp(const QModelIndex &index) {
    //qDebug() << "model MoveUp";
    
    PVRush::PVXmlTreeNodeDom *follower = nodeFromIndex(index);
    PVRush::PVXmlTreeNodeDom *parent = follower->getParent();
    PVRush::PVXmlTreeNodeDom *child;
    if (follower->getRow() >0) {
        child = parent->getChild(index.row() -1);
        //node in dom 
        parent->getDom().removeChild(follower->getDom());
        parent->getDom().insertBefore(follower->getDom(), child->getDom());
        //node in tree
        parent->removeChild(follower);
        parent->addChildAt(follower, child->getRow() );
    }
    emit layoutChanged();
 
}

/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::selectNext
 *
 *****************************************************************************/
QModelIndex PVInspector::PVXmlDomModel::selectNext(const QModelIndex &index){
    QModelIndex newSel;
    if(index.isValid()){
      if(!nodeFromIndex(index)->isOnRoot){
	int fieldCount = nodeFromIndex(index)->getParent()->getParent()->countChildren();
	int idFieldCurrent = nodeFromIndex(index)->getParent()->getRow();
	int childCount = nodeFromIndex(index)->getParent()->countChildren();
	int idChildCurrent = nodeFromIndex(index)->getRow();
	if(idChildCurrent+1<childCount){//if it isn't the last item of the field...
		newSel = index.parent().child(idChildCurrent+1,0);
	}else{//if it's the last item of the field...
		if(idFieldCurrent+1 < fieldCount){//if the node isn't on the last field...
			newSel = index.parent().parent().child(idFieldCurrent+1,0).child(0,0);
		}else{//if the node is on the last field...
			newSel = index.parent().parent().child(0,0).child(0,0);
		}
	}
      }else{
	return newSel;
      }
    }
    return newSel;//return the index to select.
}


/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::openXml
 *
 *****************************************************************************/
bool PVInspector::PVXmlDomModel::openXml(QString url) {
    //qDebug() << "PVXmlDomModel::openXml()";
    this->urlXml = url;
    QFile fichier(this->urlXml);
    if(!fichier.exists()){
        //the file doesn't exists
        QMessageBox qb;
        QString s;
        s.push_back("The file ");
        s.push_back(this->urlXml);
        s.push_back(" doesn't exists.");
        qb.setText(s);
        qb.exec();
		return false;
    }
    
    //load XML
    if (!fichier.open(QIODevice::ReadOnly)) {
        QString s;
        s.push_back("File can't be open");
        s.push_back(this->urlXml);
        message(s);
		return false;
    }
    QTextStream tmpTextXml(&fichier);
	tmpTextXml.setCodec("UTF-8"); // AG: as defined in the XML header (and saved, cf. saveXML)
	QString err_msg;
	int err_line, err_col;
	if (!this->xmlFile.setContent(tmpTextXml.readAll(), false, &err_msg, &err_line, &err_col)) {
		QMessageBox msg(QMessageBox::Critical, tr("Unable to open format"), tr("Unable to open format '%1'").arg(url), QMessageBox::Ok);
		msg.setInformativeText(QString("XML parsing error at line %1 and column %2: ").arg(err_line).arg(err_col) + err_msg);
		msg.exec();
		return false;
	}

	openXml(this->xmlFile);
	return true;
}

void PVInspector::PVXmlDomModel::openXml(QDomDocument& doc)
{
	PVRush::PVFormatVersion::to_current(doc);
	xmlRootDom = doc.documentElement();
	PVRush::PVXmlTreeNodeDom *m_rootNode = new PVRush::PVXmlTreeNodeDom(PVRush::PVXmlTreeNodeDom::field, "root", xmlRootDom, this->xmlFile);
//	if (getVersion() == "0") {
//		m_rootNode->version0to1();
//	}
	setRoot(m_rootNode);

	// Go through the DOM to get all the different groups
	m_rootNode->getGroupsByType(_groups);
#if 0
	PVRush::types_groups_t::const_iterator it;
	for (it = _groups.begin(); it != _groups.end(); it++) {
		PVLOG_INFO("type: %s, groups: ", qPrintable(it.key()));
		QSet<QString> const& grps = it.value();
		QSet<QString>::const_iterator it_g;
		for (it_g = grps.begin(); it_g != grps.end(); it_g++) {
			PVLOG_PLAIN("%s ", qPrintable(*it_g));
		}
		PVLOG_PLAIN("\n");
	}
#endif

	emit layoutChanged(); // to resfresh screen
}




/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::addUrlIn
 *
 *****************************************************************************/
void PVInspector::PVXmlDomModel::addUrlIn(const QModelIndex &index){

	PVRush::PVXmlTreeNodeDom* child;
	PVRush::PVXmlTreeNodeDom *field;
	if(index.isValid()){//if a item is selected...
		field = nodeFromIndex(index);
		if(field->typeToString() != "field"){// and if it's not a field
			message("You must select a field first.")
				return;
		}
	}else{//if no item is selected...
		field = rootNode;//current node is the root node
	}

	//conflicts Splitter & url
	if (!trustConfictSplitAxes(index))return;

	//dom
	QDomElement newDom = xmlFile.createElement("splitter");
	newDom.setAttribute("type", "url");
	QDomElement f1 = xmlFile.createElement("field");
	QDomElement f2 = xmlFile.createElement("field");
	QDomElement f3 = xmlFile.createElement("field");
	QDomElement f4 = xmlFile.createElement("field");
	QDomElement f5 = xmlFile.createElement("field");
	QDomElement f6 = xmlFile.createElement("field");

	//create axis
	field->getDom().appendChild(newDom);
	QDomElement protocol = xmlFile.createElement("axis");
	QDomElement domain = xmlFile.createElement("axis");
	QDomElement tld = xmlFile.createElement("axis");
	QDomElement port = xmlFile.createElement("axis");
	QDomElement url = xmlFile.createElement("axis");
	QDomElement variable = xmlFile.createElement("axis");


	//set all axis
	protocol.setAttribute("name", "Protocol");
	protocol.setAttribute("type", "enum");
	protocol.setAttribute("mapping", "default");
	protocol.setAttribute("plotting", "default");
	protocol.setAttribute("time-format", "");
	protocol.setAttribute("key", "false");
	protocol.setAttribute("color", "#ffffff");
	protocol.setAttribute("titlecolor", "#ffffff");
	protocol.setAttribute("tag", "protocol");
	domain.setAttribute("name", "Domain");
	domain.setAttribute("type", "host");
	domain.setAttribute("mapping", "default");
	domain.setAttribute("plotting", "default");
	domain.setAttribute("time-format", "");
	domain.setAttribute("key", "true");
	domain.setAttribute("color", "#ffffff");
	domain.setAttribute("titlecolor", "#ffffff");
	domain.setAttribute("tag", "domain");
	tld.setAttribute("name", "TLD");
	tld.setAttribute("type", "enum");
	tld.setAttribute("mapping", "default");
	tld.setAttribute("plotting", "default");
	tld.setAttribute("time-format", "");
	tld.setAttribute("key", "false");
	tld.setAttribute("color", "#ffffff");
	tld.setAttribute("titlecolor", "#ffffff");
	tld.setAttribute("tag", "tld");
	port.setAttribute("name", "Port");
	port.setAttribute("type", "integer");
	port.setAttribute("mapping", "default");
	port.setAttribute("plotting", "port");
	port.setAttribute("time-format", "");
	port.setAttribute("key", "false");
	port.setAttribute("color", "#ffffff");
	port.setAttribute("titlecolor", "#ffffff");
	port.setAttribute("tag", "port");
	url.setAttribute("name", "URL");
	url.setAttribute("type", "string");
	url.setAttribute("mapping", "default");
	url.setAttribute("plotting", "minmax");
	url.setAttribute("time-format", "");
	url.setAttribute("key", "false");
	url.setAttribute("color", "#ffffff");
	url.setAttribute("titlecolor", "#ffffff");
	url.setAttribute("tag", "url");
	variable.setAttribute("name", "Variable");
	variable.setAttribute("type", "string");
	variable.setAttribute("mapping", "default");
	variable.setAttribute("plotting", "minmax");
	variable.setAttribute("time-format", "");
	variable.setAttribute("key", "false");
	variable.setAttribute("color", "#ffffff");
	variable.setAttribute("titlecolor", "#ffffff");
	variable.setAttribute("tag", "url-variables");

	//add all axis
	f1.appendChild(protocol);
	f2.appendChild(domain);
	f3.appendChild(tld);
	f4.appendChild(port);
	f5.appendChild(url);
	f6.appendChild(variable);

	//add all fields with axis associate
	newDom.appendChild(f1);
	newDom.appendChild(f2);
	newDom.appendChild(f3);
	newDom.appendChild(f4);
	newDom.appendChild(f5);
	newDom.appendChild(f6);



	//tree
	child = new PVRush::PVXmlTreeNodeDom(newDom);
	child->setParent(field);
	field->addChild(child);


	emit layoutChanged();
}


/******************************************************************************
 *
 * PVInspector::PVXmlDomModel::trustConfictSplitAxes
 *
 *****************************************************************************/
bool PVInspector::PVXmlDomModel::trustConfictSplitAxes(const QModelIndex &index)
{
	PVRush::PVXmlTreeNodeDom* node;
	if (index.isValid()) {
		node = nodeFromIndex(index);
	}
	else {
		node = rootNode;
	}

	for (int i = 0; i < node->getChildren().count(); i++) {
		QDomElement child = node->getChildren().at(i)->getDom();
		if (child.tagName() == "axis" || child.tagName() == "RegEx" || child.tagName() == "url" || child.tagName() == "splitter") {
			message("There is just one axis or splitter in a field. Delete this one to add a newer.");
			return false;
		}
	}
	
	return true;
}

void PVInspector::PVXmlDomModel::processChildrenWithField(PVCore::PVField const& field)
{
	getRoot()->getChildrenFromField(field);
	emit layoutChanged();
}

void PVInspector::PVXmlDomModel::clearFiltersData()
{
	getRoot()->clearFiltersData();
}

void PVInspector::PVXmlDomModel::updateFiltersDataDisplay()
{
	getRoot()->updateFiltersDataDisplay();
}

void PVInspector::PVXmlDomModel::updateFieldsLinearId()
{
	getRoot()->updateFieldLinearId(0);
}

void PVInspector::PVXmlDomModel::setAxesNames(QStringList const& names)
{
	getRoot()->setAxesNames(names, 0);
}
