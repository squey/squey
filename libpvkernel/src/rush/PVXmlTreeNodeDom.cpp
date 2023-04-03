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

#include <pvkernel/rush/PVXmlTreeNodeDom.h>
#include <pvkernel/rush/PVFormat_types.h>

#define dbg                                                                                        \
	{                                                                                              \
		qDebug() << __FILE__ << ":" << __LINE__;                                                   \
	}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::PVXmlTreeNodeDom
 *
 *****************************************************************************/
PVRush::PVXmlTreeNodeDom::PVXmlTreeNodeDom()
{
	setObjectName("PVXmlTreeNodeDom");
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::PVXmlTreeNodeDom
 *
 *****************************************************************************/
PVRush::PVXmlTreeNodeDom::PVXmlTreeNodeDom(Type _type,
                                           const QString& _str,
                                           QDomElement& dom,
                                           QDomDocument& xmlFile_)
{
	init(_type, _str, dom, xmlFile_);
}

void PVRush::PVXmlTreeNodeDom::init(Type _type,
                                    const QString& _str,
                                    QDomElement& dom,
                                    QDomDocument& xmlFile_)
{
	this->type = _type;
	this->str = _str;
	this->xmlDomElement = dom;
	this->xmlFile = xmlFile_;
	parent = nullptr;
	isAlreadyExplored = false;
	isOnRoot = false;
	_field_linear_id = PVCol(-1);
}

PVRush::PVXmlTreeNodeDom* PVRush::PVXmlTreeNodeDom::new_format(QDomDocument& file)
{
	file.createProcessingInstruction("xml", "version=\"1.0\" encoding=\"utf-8\"");
	QString header(PVXmlTreeNodeDom_initXml);
	QString err;

	file.setContent(header, true, &err);
	QDomElement xmlRootDom = file.documentElement();
	xmlRootDom.setAttribute("version", PVFORMAT_CURRENT_VERSION);

	// creating the root node.
	auto* rootNode = new PVRush::PVXmlTreeNodeDom(
	    PVRush::PVXmlTreeNodeDom::Type::field, "root", xmlRootDom, file);
	return rootNode;
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::PVXmlTreeNodeDom
 *
 *****************************************************************************/
PVRush::PVXmlTreeNodeDom::PVXmlTreeNodeDom(QDomElement const& dom)
{
	setTypeFromString(dom.tagName());
	this->xmlDomElement = dom;
	parent = nullptr;
	isAlreadyExplored = false;
	isOnRoot = false;
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::~PVXmlTreeNodeDom
 *
 *****************************************************************************/
PVRush::PVXmlTreeNodeDom::~PVXmlTreeNodeDom()
{
	qDeleteAll(children);
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::addChild
 *
 *****************************************************************************/
void PVRush::PVXmlTreeNodeDom::addChild(PVRush::PVXmlTreeNodeDom* child)
{
	PVLOG_DEBUG("PVRush::PVXmlTreeNodeDom::addChild\n");
	child->setParent(this); // set his parent
	child->setDoc(xmlFile); // set the dom node
	if (child->xmlDomElement.tagName() == "axis" || child->xmlDomElement.tagName() == "RegEx" ||
	    child->xmlDomElement.tagName() == "splitter" || child->xmlDomElement.tagName() == "url" ||
	    child->xmlDomElement.tagName() == "field") {
		this->children.push_back(child); // child adding (in last position for axis, regex and url
	} else
		this->children.push_front(child); // child adding (in first for filter)
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::addChildAt
 *
 *****************************************************************************/
void PVRush::PVXmlTreeNodeDom::addChildAt(PVRush::PVXmlTreeNodeDom* child, int row)
{
	this->children.insert(row, child);
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::getChild
 *
 *****************************************************************************/
PVRush::PVXmlTreeNodeDom* PVRush::PVXmlTreeNodeDom::getChild(int i)
{
	return this->getChildren().takeAt(i);
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::getChildren()
 *
 *****************************************************************************/
QList<PVRush::PVXmlTreeNodeDom*> PVRush::PVXmlTreeNodeDom::getChildren()
{
	if (this->isAlreadyExplored)
		return this->children;
	else {
		explore();
	} // explore the tree structure.
	return this->children;
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::countChildren
 *
 *****************************************************************************/
int PVRush::PVXmlTreeNodeDom::countChildren()
{
	return this->getChildren().count();
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::createSplitterPlugin
 *
 *****************************************************************************/
bool PVRush::PVXmlTreeNodeDom::createSplitterPlugin(const QDomElement& domElt)
{
	QString plugName = domElt.attribute("type", "-");
	PVFilter::PVFieldsSplitterParamWidget_p in_t =
	    LIB_CLASS(PVFilter::PVFieldsSplitterParamWidget)::get().get_class_by_name(plugName);
	if (!in_t) {
		return false;
	}
	PVFilter::PVFieldsSplitterParamWidget_p in_t_cpy =
	    in_t->clone<PVFilter::PVFieldsSplitterParamWidget>();
	QString registered_name = in_t_cpy->registered_name();
	setSplitterPlugin(in_t_cpy);
	return true;
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::createConverterPlugin
 *
 *****************************************************************************/
bool PVRush::PVXmlTreeNodeDom::createConverterPlugin(const QDomElement& domElt)
{
	QString plugName = domElt.attribute("type", "-");
	PVFilter::PVFieldsConverterParamWidget_p in_t =
	    LIB_CLASS(PVFilter::PVFieldsConverterParamWidget)::get().get_class_by_name(plugName);
	if (!in_t) {
		return false;
	}
	PVFilter::PVFieldsConverterParamWidget_p in_t_cpy =
	    in_t->clone<PVFilter::PVFieldsConverterParamWidget>();
	QString registered_name = in_t_cpy->registered_name();
	setConverterPlugin(in_t_cpy);
	return true;
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::getParent
 *
 *****************************************************************************/
PVRush::PVXmlTreeNodeDom* PVRush::PVXmlTreeNodeDom::getParent()
{
	return this->parent;
}

/******************************************************************************
 *
 *  PVRush::PVXmlTreeNodeDom::setParent
 *
 *****************************************************************************/
void PVRush::PVXmlTreeNodeDom::setParent(PVRush::PVXmlTreeNodeDom* p_parent)
{
	this->parent = p_parent;
}

/******************************************************************************
 *
 * PVInspector::PVXmlTreeNodeDom::getName
 *
 *****************************************************************************/
QString PVRush::PVXmlTreeNodeDom::getName()
{
	// if(this->type==Root)return "root";
	switch (this->type) {
	case Type::RegEx:
	case Type::axis:
	case Type::url:
	case Type::filter:
		return this->xmlDomElement.attribute(PVFORMAT_FILTER_NAME_STR,
		                                     PVFORMAT_FILTER_NAME_DEFAULT);
	case Type::field:
		if (isFieldOfUrl()) {   // if node is an url...
			switch (getRow()) { //...return name as following
			case 0:
				return "protocol";
			case 1:
				return "domain";
			case 2:
				return "tld";
			case 3:
				return "port";
			case 4:
				return "url";
			case 5:
				return "variable";
			}
			return "url field";
		} else { //... else get the attribute "name".
			QString selectionRegExpName = QString("selectionRegExp-%0").arg(getRow() + 1);
			QString selectionRegExp = getParent()->attribute(selectionRegExpName, "");
			return selectionRegExp;
		}
	case Type::splitter:
		return this->xmlDomElement.attribute(PVFORMAT_FILTER_NAME_STR,
		                                     PVFORMAT_FILTER_NAME_DEFAULT);
	default:
		return this->str;
	}
	return this->str;
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::setName
 *
 *****************************************************************************/
void PVRush::PVXmlTreeNodeDom::setName(QString nom)
{
	switch (this->type) {
	case Type::RegEx: // // set the attribute name
	case Type::axis:  // // set the attribute name
	case Type::filter:
	case Type::splitter:
		this->xmlDomElement.setAttribute(PVFORMAT_FILTER_NAME_STR, nom);
		break;
	case Type::field:
	default:
		this->str = nom;
	}
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::explore
 *
 *****************************************************************************/
void PVRush::PVXmlTreeNodeDom::explore()
{
	PVLOG_DEBUG("PVRush::PVXmlTreeNodeDom::explore\n");
	this->isAlreadyExplored = true;

	/* search for child in the dom.*/
	QDomElement childItem = this->xmlDomElement.firstChildElement();

	while (!childItem.isNull()) {
		QString tag_name(childItem.tagName());
		if (tag_name != PVFORMAT_XML_TAG_MAPPING && tag_name != PVFORMAT_XML_TAG_PLOTTING) {
			addChild(new PVXmlTreeNodeDom(childItem));
		}
		childItem = childItem.nextSiblingElement();
	}
	PVLOG_DEBUG("PVRush::PVXmlTreeNodeDom::explore : end\n");
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::setTypeFromString
 *
 *****************************************************************************/
void PVRush::PVXmlTreeNodeDom::setTypeFromString(const QString& nom)
{
	if (nom == QString("RegEx"))
		type = Type::RegEx;
	else if (nom == QString("filter"))
		type = Type::filter;
	else if (nom == QString("field"))
		type = Type::field;
	else if (nom == QString("axis"))
		type = Type::axis;
	else if (nom == QString("url"))
		type = Type::url;
	else if (nom == QString("splitter"))
		type = Type::splitter;
	else if (nom == QString("converter"))
		type = Type::converter;
	else
		PVLOG_ERROR("in PVRush::PVXmlTreeNodeDom::setTypeFromString : type unknow\n");
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::deleteFromTree
 *
 *****************************************************************************/
void PVRush::PVXmlTreeNodeDom::deleteFromTree()
{
	if (this->getParent()->getDom().tagName() != "param") {
		this->xmlDomElement.parentNode().removeChild(this->xmlDomElement);
		this->getParent()->removeChild(this);
	} else {
		this->getParent()->getDom().removeChild(this->xmlDomElement);
		this->getParent()->removeChild(this);
	}
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::removeChild
 *
 *****************************************************************************/
void PVRush::PVXmlTreeNodeDom::removeChild(PVXmlTreeNodeDom* child)
{
	this->children.removeOne(child);
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::getRow
 *
 *****************************************************************************/
int PVRush::PVXmlTreeNodeDom::getRow()
{
	return parent->children.indexOf(this, 0);
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::setDoc
 *
 *****************************************************************************/
void PVRush::PVXmlTreeNodeDom::setDoc(QDomDocument& file)
{
	this->xmlFile = file;
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::getExpression
 *
 *****************************************************************************/
QString PVRush::PVXmlTreeNodeDom::getExpression()
{
	switch (this->type) {
	case Type::RegEx:;
	case Type::filter:
		return this->xmlDomElement.attribute("expression", ".*");
	default:
		break;
	}
	return "";
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::setExpression
 *
 *****************************************************************************/
void PVRush::PVXmlTreeNodeDom::setExpression(QString exp)
{
	switch (this->type) {
	case Type::RegEx:
	case Type::filter:
		this->xmlDomElement.setAttribute("expression", exp);
		break;
	case Type::field:
	default:
		break;
	}
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::getNbr
 *
 *****************************************************************************/
int PVRush::PVXmlTreeNodeDom::getNbr()
{
	switch (this->type) {
	case Type::RegEx:
		return this->xmlDomElement.childNodes().count();
	case Type::splitter:
		return this->xmlDomElement.childNodes().count();
	case Type::converter:
		return this->xmlDomElement.childNodes().count();
	default:
		break;
	}
	return 0;
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::getOutName
 *
 *****************************************************************************/
QString PVRush::PVXmlTreeNodeDom::getOutName()
{
	QString l;
	if (this->type == Type::field) { // its a field whiche is selected...
		for (int i = 0; i < children.count(); i++) {
			if (children.at(i)->getDom().tagName() == PVFORMAT_XML_TAG_AXIS_STR)
				return children.at(i)->getDom().attribute(
				    PVFORMAT_AXIS_NAME_STR,
				    PVFORMAT_AXIS_NAME_DEFAULT); // return the attribute name.
			if (children.at(i)->getDom().tagName() == "url")
				return "URL";
			if (children.at(i)->getDom().tagName() == "RegEx")
				return children.at(i)->getDom().attribute("name", " ");
		}
	}
	return l;
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::getOutWidget
 *
 *****************************************************************************/
PVRush::PVXmlTreeNodeDom* PVRush::PVXmlTreeNodeDom::getOutWidget()
{
	PVXmlTreeNodeDom* l = nullptr;
	if (this->type == Type::field) { // its a field whiche is selected...
		for (int i = 0; i < children.count(); i++) {
			if ((children.at(i)->getDom().tagName() ==
			     PVFORMAT_XML_TAG_AXIS_STR) ||                     // if it's an axis
			    (children.at(i)->getDom().tagName() == "url") ||   // if it's an url
			    (children.at(i)->getDom().tagName() == "RegEx")) { // if it's a regexp
				return children.at(i);                             // return the node
			}
		}
	}
	return l;
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::setNbr
 *
 *****************************************************************************/
void PVRush::PVXmlTreeNodeDom::setNbr(int nbr)
{
	// this function update the number of field in the regexp
	switch (this->type) {
	case Type::RegEx:
	case Type::splitter:
		PVLOG_DEBUG("PVRush::PVXmlTreeNodeDom::setNbr(%d)\n", nbr);
		if (nbr > this->getNbr()) {
			addField(nbr - this->getNbr()); // add some fields
		} else if (nbr < this->getNbr()) {
			delField(this->getNbr() - nbr); // delete some fiels
		}
		break;
	case Type::converter:
		PVLOG_DEBUG("PVRush::PVXmlTreeNodeDom::setNbr(%d)\n", nbr);
		if (this->getNbr() == 0) {
			addField(1);
		}
		break;
	default:
		break;
	}
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::getDom
 *
 *****************************************************************************/
QDomElement PVRush::PVXmlTreeNodeDom::getDom()
{
	return xmlDomElement;
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::addField
 *
 *****************************************************************************/
void PVRush::PVXmlTreeNodeDom::addField(int nbr)
{
	PVLOG_DEBUG("PVRush::PVXmlTreeNodeDom::addField(%d)\n", nbr);
	if ((this->type == Type::RegEx || this->type == Type::splitter ||
	     this->type == Type::converter) &&
	    nbr > 0) {
		for (int i = 0; i < nbr; i++) {
			addOneField("");
		}
	}
}

PVRush::PVXmlTreeNodeDom* PVRush::PVXmlTreeNodeDom::addOneField(QString const& name)
{
	// dom field
	QDomElement newField = this->xmlFile.createElement("field");
	xmlDomElement.appendChild(newField); // Set its parent
	// dom axis
	QDomElement newAxisDom = this->xmlFile.createElement("axis");
	newField.appendChild(newAxisDom); // set its parent as the field

	// node field
	auto newNodeField = new PVXmlTreeNodeDom(newField); // création
	newNodeField->setParent(this);                      // modif du parent
	// node axis (we add an axis in each field)
	auto newAxis = new PVXmlTreeNodeDom(newAxisDom);
	newAxis->setParent(newNodeField);
	newAxis->setName(name);

	this->children.push_back(newNodeField); // Put it in the view

	return newAxis;
}

PVRush::PVXmlTreeNodeDom* PVRush::PVXmlTreeNodeDom::addOneField(QString const& name,
                                                                QString const& axis_type)
{
	PVXmlTreeNodeDom* axis = addOneField(name);
	axis->setAttribute("type", axis_type);
	return axis;
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::delField
 *
 *****************************************************************************/

/**
 * delete the n last field.
 * @param n
 */
void PVRush::PVXmlTreeNodeDom::delField(int n)
{
	PVLOG_DEBUG("PVRush::PVXmlTreeNodeDom::delField(%d)\n", n);
	for (int i = 0; i < n; i++) {
		PVXmlTreeNodeDom* lastChild = children.at(children.count() - 1);
		removeChild(lastChild);
		getDom().removeChild(lastChild->getDom());
	}
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::addFilterRacine
 *
 *****************************************************************************/
void PVRush::PVXmlTreeNodeDom::addFilterRacine()
{
	QDomElement elt = this->xmlFile.createElement("filter"); // create the dom element
	auto child = new PVXmlTreeNodeDom(elt);                  // create the node
	child->isOnRoot = true;
	child->setParent(this); // place the node
	this->addChild(child);  //...
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::typeToString
 *
 *****************************************************************************/
QString PVRush::PVXmlTreeNodeDom::typeToString()
{
	// return the type as a QString
	if (type == Type::RegEx)
		return "RegEx";
	if (type == Type::filter)
		return "filter";
	if (type == Type::field)
		return "field";
	if (type == Type::axis)
		return "axis";
	if (type == Type::Root)
		return "Root";
	if (type == Type::url)
		return "url";
	if (type == Type::splitter)
		return "splitter";
	if (type == Type::converter)
		return "converter";
	return "";
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::setAttribute
 *
 *****************************************************************************/
void PVRush::PVXmlTreeNodeDom::setAttribute(QString name, QString value, bool flagSaveInXml)
{
	// PVLOG_INFO("PVXmlTreeNodeDom::setAttribute('%s','%s')\n",name.toUtf8().constData(),value.toUtf8().constData());
	if (flagSaveInXml) {
		xmlDomElement.setAttribute(name, value);
	} else {
		xmlDomElement.setAttribute(name, QString(""));
	}
	otherData.insert(name, value);
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::getAttribute
 *
 *****************************************************************************/
QString PVRush::PVXmlTreeNodeDom::attribute(QString name, bool flagReadInXml)
{
	// PVLOG_INFO("PVXmlTreeNodeDom::getAttribute( %s )\n",name.toUtf8().constData());
	if (flagReadInXml) {
		// update the savedvalue
		if (otherData.contains(name)) {
			otherData[name] = xmlDomElement.attribute(name, "");
		} else {
			otherData.insert(name, xmlDomElement.attribute(name, ""));
		}
		// read the dom element
		return xmlDomElement.attribute(name, "");
	} else if (otherData.contains(name)) {
		return otherData[name];
	}

	return {};
}

/******************************************************************************
 *
 * PVRush::PVXmlTreeNodeDom::isFieldOfUrl
 *
 *****************************************************************************/
bool PVRush::PVXmlTreeNodeDom::isFieldOfUrl()
{
	return (getParent()->xmlDomElement.tagName() == "url");
}

void PVRush::PVXmlTreeNodeDom::setFromArgumentList(QDomElement& elt,
                                                   PVCore::PVArgumentList const& def_args,
                                                   PVCore::PVArgumentList const& args)
{
	PVCore::PVArgumentList::const_iterator it;
	for (it = args.begin(); it != args.end(); it++) {
		QString key = it->key();
		QVariant v = it->value();
		QVariant def_v = def_args.at(key);
		if (v != def_v) {
			elt.setAttribute(key, PVCore::PVArgument_to_QString(v));
		}
	}
}

void PVRush::PVXmlTreeNodeDom::setFromArgumentList(PVCore::PVArgumentList const& args)
{
	PVCore::PVArgumentList::const_iterator it;
	for (it = args.begin(); it != args.end(); it++) {
		setAttribute(it->key(), PVCore::PVArgument_to_QString(it->value()), true);
	}
}

void PVRush::PVXmlTreeNodeDom::toArgumentList(QDomElement& elt,
                                              PVCore::PVArgumentList const& def_args,
                                              PVCore::PVArgumentList& args)
{
	PVCore::PVArgumentList::const_iterator it;
	for (it = def_args.begin(); it != def_args.end(); it++) {
		QString const& key = it->key();
		QString v = elt.attribute(key);
		PVCore::PVArgument vset;
		if (v.trimmed().isEmpty()) {
			vset = it->value();
		} else {
			vset = PVCore::QString_to_PVArgument(v, it->value());
		}
		args[key] = vset;
	}
}

void PVRush::PVXmlTreeNodeDom::toArgumentList(PVCore::PVArgumentList const& default_args,
                                              PVCore::PVArgumentList& args)
{
	PVCore::PVArgumentList::const_iterator it;
	for (it = default_args.begin(); it != default_args.end(); it++) {
		QString const& key = it->key();
		QString v = attribute(key, true);
		PVCore::PVArgument vset;
		if (v.isEmpty()) {
			vset = it->value();
		} else {
			vset = PVCore::QString_to_PVArgument(v, it->value());
		}
		args[key] = vset;
	}
}

void PVRush::PVXmlTreeNodeDom::getChildrenFromField(PVCore::PVField const& field_)
{
	QString plugin_type = getDom().tagName();
	// TODO: filters (like filter_regexp) should also be ok
	if (plugin_type != "splitter" && plugin_type != "converter") {
		// This is not a splitter, so pass this through our children.
		for (int i = 0; i < getChildren().size(); i++) {
			getChild(i)->getChildrenFromField(field_);
		}
		return;
	}

	PVCore::PVField field(field_);
	QString f_value = QString::fromUtf8(field.begin(), field.size());

	QString plugin_name = attribute("type", "");

	// Get the filter from the lib (because not everything is under plugins... :/)
	PVFilter::PVFieldsSplitter_p filter_lib =
	    LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_class_by_name(plugin_name);
	if (!filter_lib) {
		PVLOG_ERROR("Unable to load splitter plugin '%s' !!!!!!\n", qPrintable(plugin_name));
		return;
	}
	PVFilter::PVFieldsSplitter_p filter_clone;
	filter_clone = filter_lib->clone<PVFilter::PVFieldsSplitter>();

	// Get the args and set them
	PVCore::PVArgumentList args;
	toArgumentList(filter_lib->get_default_args(), args);
	filter_clone->set_args(args);

	// Set the number of expected children
	filter_clone->set_number_expected_fields(countChildren());

	// Check if a number of children is forced
	size_t force_nchild = 0;
	// TODO: this should be all in plugins !
	if (splitterPlugin) {
		force_nchild = splitterPlugin->force_number_children();
		splitterPlugin->push_data(f_value);
	} else if (converterPlugin) {
		converterPlugin->push_data(f_value);
	} else if (plugin_name == "url") {
		// TODO: that's a real issue here.
		// For the URL splitter, you also need to change that number !!
		force_nchild = 10;
	}

	// Ok, we are a splitter. Process this through our filter.
	PVCore::list_fields lf;
	lf.push_back(field);
	PVCore::list_fields& lf_res = filter_clone->operator()(lf);
	if (!field.elt_parent()->valid()) {
		// The filter failed, we can't do much from now.
		// PVLOG_INFO("(getChildrenFromField) splitter returns an invalid field.\n",
		// qPrintable(plugin_name), lf_res.size());
		if (force_nchild > 0) {
			setNbr(force_nchild);
		}
		return;
	}

	// We got our number of children !
	setNbr(lf_res.size());

	// If force_nchild > 0, set this number of child
	if (force_nchild > 0) {
		setNbr(force_nchild);
	} else {
		setNbr(lf_res.size());
	}

	// If number of childs has been forced and the number of fields at the output of the filter
	// aren't the same, we just return ! (the filter failed)
	if (force_nchild > 0 && force_nchild != lf_res.size()) {
		return;
	}

	// Pass the resulting fields to the children
	// TODO: AG: ugly, I know that
	auto it_f = lf_res.begin();
	for (size_t ichild = 0; ichild < lf_res.size(); ichild++) {
		getChild(ichild)->getChildrenFromField(*it_f);
		it_f++;
		if (it_f == lf_res.end()) {
			break;
		}
	}
}

void PVRush::PVXmlTreeNodeDom::clearFiltersData()
{
	if (splitterPlugin) {
		splitterPlugin->clear_filter_data();
	} else if (converterPlugin) {
		converterPlugin->clear_filter_data();
	}

	for (int ichild = 0; ichild < getChildren().size(); ichild++) {
		getChild(ichild)->clearFiltersData();
	}
}

void PVRush::PVXmlTreeNodeDom::updateFiltersDataDisplay()
{
	if (splitterPlugin) {
		splitterPlugin->update_data_display();
	} else if (converterPlugin) {
		converterPlugin->update_data_display();
	}

	for (int ichild = 0; ichild < getChildren().size(); ichild++) {
		getChild(ichild)->updateFiltersDataDisplay();
	}
}

PVCol PVRush::PVXmlTreeNodeDom::updateFieldLinearId(PVCol id)
{
	size_t nchilds = getChildren().size();
	if (getDom().tagName() == "field") {
		if (hasAxisAsChild()) {
			_field_linear_id = id;
			id++;
		} else {
			_field_linear_id = PVCol(-1);
		}
	}

	for (size_t ichild = 0; ichild < nchilds; ichild++) {
		id = getChild(ichild)->updateFieldLinearId(id);
	}

	// Return the id of the next field
	return id;
}

bool PVRush::PVXmlTreeNodeDom::hasAxisAsChild()
{
	for (const auto node : getChildren()) {
		if (node->typeToString() == PVFORMAT_XML_TAG_AXIS_STR) {
			return true;
		}
	}

	return false;
}

PVRush::PVXmlTreeNodeDom* PVRush::PVXmlTreeNodeDom::getFirstFieldParent()
{
	PVXmlTreeNodeDom* parent = getParent();
	if (parent == nullptr) {
		// No more parent, so we can't find any parent field. Returns nullptr
		return nullptr;
	}

	if (parent->typeToString() == "field" && parent->_field_linear_id != -1) {
		// We got it !
		return parent;
	}

	// Go & see what our parent has to say about this !
	return parent->getFirstFieldParent();
}

PVCol PVRush::PVXmlTreeNodeDom::setAxesNames(QStringList const& names, PVCol id)
{
	if (type == Type::axis) {
		// Set its name
		if (id < names.size()) {
			setName(names[id]);
		} else {
			PVLOG_WARN("(PVXmlTreeNodeDom::setAxesNames) not enough names given (axis id = %d, "
			           "size of list = %d).\n",
			           id, names.size());
			return id;
		}
	} else if (type == Type::field) {
		for (auto* child : getChildren()) {
			id = child->setAxesNames(names, id);
		}
		++id;
	} else if (type == Type::splitter) {
		for (auto* child : getChildren()) {
			id = child->setAxesNames(names, id);
		}
	}

	return id;
}

QDomElement PVRush::PVXmlTreeNodeDom::getMappingElement()
{
	if (type != Type::axis) {
		return {};
	}

	QDomNodeList elts_mapping = getDom().elementsByTagName(PVFORMAT_XML_TAG_MAPPING);
	if (elts_mapping.size() < 1) {
		QDomElement el_map = getDom().ownerDocument().createElement(PVFORMAT_XML_TAG_MAPPING);
		getDom().appendChild(el_map);
		return el_map;
	}

	return elts_mapping.at(0).toElement();
}

QDomElement PVRush::PVXmlTreeNodeDom::getPlottingElement()
{
	if (type != Type::axis) {
		return {};
	}

	QDomNodeList elts_mapping = getDom().elementsByTagName(PVFORMAT_XML_TAG_PLOTTING);
	if (elts_mapping.size() < 1) {
		QDomElement el_map = getDom().ownerDocument().createElement(PVFORMAT_XML_TAG_PLOTTING);
		getDom().appendChild(el_map);
		return el_map;
	}

	return elts_mapping.at(0).toElement();
}

QString PVRush::PVXmlTreeNodeDom::getMappingProperties(PVCore::PVArgumentList const& def_args,
                                                       PVCore::PVArgumentList& args)
{
	QDomElement elt = getMappingElement();
	if (elt.isNull()) {
		return {};
	}

	toArgumentList(elt, def_args, args);

	return elt.attribute(PVFORMAT_MAP_PLOT_MODE_STR);
}

QString PVRush::PVXmlTreeNodeDom::getPlottingProperties(PVCore::PVArgumentList const& def_args,
                                                        PVCore::PVArgumentList& args)
{
	QDomElement elt = getPlottingElement();
	if (elt.isNull()) {
		return {};
	}

	toArgumentList(elt, def_args, args);

	return elt.attribute(PVFORMAT_MAP_PLOT_MODE_STR);
}

void PVRush::PVXmlTreeNodeDom::deleteAllAttributes(QDomElement& elt)
{
	QDomNamedNodeMap map_attrs = elt.attributes();
	for (int i = 0; i < map_attrs.size(); i++) {
		elt.removeAttribute(map_attrs.item(i).toAttr().name());
	}
}

void PVRush::PVXmlTreeNodeDom::setMappingProperties(QString const& mode,
                                                    PVCore::PVArgumentList const& def_args,
                                                    PVCore::PVArgumentList const& args)
{
	if (type != Type::axis) {
		return;
	}
	QDomElement elt_mapping = getMappingElement();
	if (elt_mapping.isNull()) {
		return;
	}
	deleteAllAttributes(elt_mapping);
	setFromArgumentList(elt_mapping, def_args, args);
	elt_mapping.setAttribute(PVFORMAT_MAP_PLOT_MODE_STR, mode);
}

void PVRush::PVXmlTreeNodeDom::setPlottingProperties(QString const& mode,
                                                     PVCore::PVArgumentList const& def_args,
                                                     PVCore::PVArgumentList const& args)
{
	if (type != Type::axis) {
		return;
	}
	QDomElement elt_plotting = getPlottingElement();
	if (elt_plotting.isNull()) {
		return;
	}
	deleteAllAttributes(elt_plotting);
	setFromArgumentList(elt_plotting, def_args, args);
	elt_plotting.setAttribute(PVFORMAT_MAP_PLOT_MODE_STR, mode);
}
