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

#include <PVXmlDomModel.h>

#include <pvkernel/filter/PVFieldsFilterParamWidget.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVFormat_types.h>
#include <pvkernel/rush/PVFormatVersion.h>
#include <pvkernel/rush/PVXmlTreeNodeDom.h>
#include <pvkernel/core/PVUtils.h>

#include <QString>

#include <memory>
#include <numeric>

/******************************************************************************
 *
 * App::PVXmlDomModel::PVXmlDomModel
 *
 *****************************************************************************/
App::PVXmlDomModel::PVXmlDomModel(QWidget* parent)
    : QAbstractItemModel(parent), _axes_combination(_axes)
{

	PVRush::PVXmlTreeNodeDom* m_rootNode = PVRush::PVXmlTreeNodeDom::new_format(xmlFile);
	xmlRootDom = xmlFile.documentElement();
	setRoot(m_rootNode);
	setObjectName("PVXmlDomModel");

	_original_xml_content = xmlFile.toString();
}

/******************************************************************************
 *
 * App::PVXmlDomModel::~PVXmlDomModel
 *
 *****************************************************************************/
App::PVXmlDomModel::~PVXmlDomModel() = default;

/******************************************************************************
 *
 * App::PVXmlDomModel::index
 *
 *****************************************************************************/
/**
 * create a new index for the treeview.
 */
QModelIndex App::PVXmlDomModel::index(int r, int c, const QModelIndex& parent) const
{

	if (!rootNode || r < 0 || c < 0)
		return {};

	PVRush::PVXmlTreeNodeDom* parentNode = nodeFromIndex(parent);
	PVRush::PVXmlTreeNodeDom* childNode = parentNode->getChildren().value(r);

	if (!childNode)
		return {};

	return createIndex(r, c, childNode);
}

/******************************************************************************
 *
 * App::PVXmlDomModel::indexOfChild
 *
 *****************************************************************************/

QModelIndex App::PVXmlDomModel::indexOfChild(const QModelIndex& parent,
                                                     const PVRush::PVXmlTreeNodeDom* node) const
{
	QModelIndex model_index = index(0, 0, parent);
	int i = 0;

	while (model_index.isValid()) {
		if (nodeFromIndex(model_index) == node) {
			return model_index;
		}

		model_index = index(i++, 0, parent);
	}

	return {};
}

/******************************************************************************
 *
 * App::PVXmlDomModel::parent
 *
 *****************************************************************************/
/**
 * Get the parent index
 * @param child node
 */
QModelIndex App::PVXmlDomModel::parent(const QModelIndex& child) const
{

	PVRush::PVXmlTreeNodeDom* node = nodeFromIndex(child);
	if (!node)
		return {};

	PVRush::PVXmlTreeNodeDom* parentNode = node->getParent();
	if (!parentNode)
		return {};

	PVRush::PVXmlTreeNodeDom* grandParentNode = parentNode->getParent();
	if (!grandParentNode)
		return {};

	int row = grandParentNode->getChildren().indexOf(parentNode);
	return createIndex(row, 0, parentNode);
}

/******************************************************************************
 *
 * App::PVXmlDomModel::rowCount
 *
 *****************************************************************************/

int App::PVXmlDomModel::rowCount(const QModelIndex& parent) const
{

	if (parent.column() > 0)
		return 0;

	PVRush::PVXmlTreeNodeDom* parentNode = nodeFromIndex(parent);
	if (!parentNode)
		return 0;

	return parentNode->countChildren();
}

/******************************************************************************
 *
 * App::PVXmlDomModel::columnCount
 *
 *****************************************************************************/
/**
 * Return the number of raw.
 */
int App::PVXmlDomModel::columnCount(const QModelIndex&) const
{
	return 2;
}

/******************************************************************************
 *
 * App::PVXmlDomModel::countParent
 *
 *****************************************************************************/
/**
 * return the number of node from the root to the child.
 */
int App::PVXmlDomModel::countParent(const QModelIndex& index)
{
	if (index.parent().isValid() && nodeFromIndex(index)->getName() != "root") {
		return (countParent(index.parent()) + 1);
	}
	return 0;
}

/******************************************************************************
 *
 * App::PVXmlDomModel::data
 *
 *****************************************************************************/
QVariant App::PVXmlDomModel::data(const QModelIndex& index, int role) const
{

	if (index.isValid()) {
		if (role == Qt::DisplayRole || role == Qt::EditRole) {
			PVRush::PVXmlTreeNodeDom* node = nodeFromIndex(index);
			assert(node);
			if (index.column() == 0) {
				if (node->typeToString() == "RegEx") {
					return "Splitter (RegEx)";
				} else if (node->typeToString() == "url") {
					return "Splitter (URL)";
				} else if (node->typeToString() == "splitter") {
					QString s = QString("splitter : %1").arg(node->attribute("type", ""));
					return s;
				} else if (node->typeToString() == "converter") {
					QString s = QString("converter : %1").arg(node->attribute("type", ""));
					return s;
				}
				return node->typeToString();

			} else if (index.column() == 1) {
				if (node->typeToString() == "field") {
					return QString("");
				}
				// if(node->typeToString()=="RegEx")return
				// node->getAttribute(QString("expression"));
				return node->getName();
			} else {
				return QString("");
			}
		}
	} else
		return {};
	return {};
}

/******************************************************************************
 *
 * App::PVXmlDomModel::setRoot
 *
 *****************************************************************************/
/**
 * Setup the root.
 */
void App::PVXmlDomModel::setRoot(PVRush::PVXmlTreeNodeDom* node)
{
	beginResetModel();
	rootNode.reset(node);
	endResetModel();
}

/******************************************************************************
 *
 * App::PVXmlDomModel::nodeFromIndex
 *
 *****************************************************************************/
/**
 * Return the node from the index.
 * @param index
 */
PVRush::PVXmlTreeNodeDom* App::PVXmlDomModel::nodeFromIndex(const QModelIndex& index) const
{
	if (index.isValid()) {
		assert(index.internalPointer());
		// PVLOG_DEBUG("                    nodeFromIndex            %s
		// \n",qPrintable(static_cast<PVRush::PVXmlTreeNodeDom
		// *>(index.internalPointer())->getName()));
		// PVLOG_DEBUG("              ---      \n");
		return static_cast<PVRush::PVXmlTreeNodeDom*>(index.internalPointer());
	} else {
		return getRoot();
	}
}

/******************************************************************************
 *
 * App::PVXmlDomModel::flags
 *
 *****************************************************************************/
Qt::ItemFlags App::PVXmlDomModel::flags(const QModelIndex& index) const
{
	Qt::ItemFlags flags = Qt::ItemIsSelectable | Qt::ItemIsEnabled;
	if (index.column() == 1) {
		if (nodeFromIndex(index)->isEditable()) {
			flags = flags | Qt::ItemIsEditable;
		}
	}
	return flags;
}

/******************************************************************************
 *
 * App::PVXmlDomModel::headerData
 *
 *****************************************************************************/
QVariant
App::PVXmlDomModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	// If it's view is displaying
	if (role == Qt::DisplayRole) {
		// if the orientation is horizontal
		if (orientation == Qt::Horizontal) {
			switch (section) {
			case 0: // edit first column
				return {"Type"};
				break;
			case 1: // edit second column
				return {"Name"};
				break;
			}
		}
	}
	return {};
}

/******************************************************************************
 *
 * App::PVXmlDomModel::supportedDropActions
 *
 *****************************************************************************/
Qt::DropActions App::PVXmlDomModel::supportedDropActions() const
{
	return Qt::CopyAction | Qt::MoveAction;
}

/******************************************************************************
 *
 * App::PVXmlDomModel::setData
 *
 *****************************************************************************/
bool App::PVXmlDomModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
	if (role == Qt::EditRole) {
		if (index.column() == 1) { // just choose the second column
			nodeFromIndex(index)->setName(value.toString());
			Q_EMIT dataChanged(index, index);
			return true;
		}
	}
	return false;
}

/******************************************************************************
 *
 * App::PVXmlDomModel::saveXml
 *
 *****************************************************************************/
bool App::PVXmlDomModel::saveXml(QString xml_file)
{

	if (!xml_file.endsWith(".format")) {
		xml_file.append(".format");
	}

	QFile file(xml_file);
	if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
		QMessageBox msg(QMessageBox::Critical, "Error while saving format",
		                QString("Unable to open %1 for writing: ").arg(xml_file) +
		                    file.errorString(),
		                QMessageBox::Ok);
		msg.exec();
		return false;
	}
	QString version = QString("%1").arg(PVFORMAT_CURRENT_VERSION);
	xmlRootDom.setAttribute("version", version);

	// Add the axes-combination
	updateAxesCombination();
	if (not _axes_combination.get_combination().empty()) {
		QDomElement axis_comb_elt = xmlFile.createElement(PVFORMAT_XML_TAG_AXES_COMBINATION_STR);
		QDomText axis_comb_txt = xmlFile.createTextNode(_axes_combination.to_string());
		axis_comb_elt.appendChild(axis_comb_txt);
		xmlRootDom.appendChild(axis_comb_elt);
	}
	QByteArray data(xmlFile.toString().toUtf8());

	int size_written = file.write(data);
	int data_size = data.size();
	if (size_written != data_size || !file.flush()) {
		QMessageBox msg(QMessageBox::Critical, "Error while saving format",
		                QString("An error occured while saving format: ") + file.errorString(),
		                QMessageBox::Ok);
		msg.exec();
		return false;
	}
	file.close();

	// Remove the axis-combination tag
	QDomElement axes_cb_elt = xmlRootDom.firstChildElement(PVFORMAT_XML_TAG_AXES_COMBINATION_STR);
	if (!axes_cb_elt.isNull()) {
		xmlRootDom.removeChild(axes_cb_elt);
	}

	_original_xml_content = xmlFile.toString();

	return true;
}

/******************************************************************************
 *
 * App::PVXmlDomModel::applyModification
 *
 *****************************************************************************/
void App::PVXmlDomModel::applyModification(QModelIndex&, PVXmlParamWidget*)
{
	Q_EMIT layoutChanged();
}

/******************************************************************************
 *
 * App::PVXmlDomModel::deleteSelectio
 *
 *****************************************************************************/
void App::PVXmlDomModel::deleteSelection(QModelIndex const& index)
{
	if (index.isValid()) {
		PVRush::PVXmlTreeNodeDom* nodeASupprimer = nodeFromIndex(index);
		if (nodeASupprimer != rootNode.get()) {
			nodeASupprimer->deleteFromTree();
			Q_EMIT layoutChanged();
		}
	} else {
	}
}

/******************************************************************************
 *
 * App::PVXmlDomModel::addAxisIn
 *
 *****************************************************************************/
PVRush::PVXmlTreeNodeDom* App::PVXmlDomModel::addAxisIn(const QModelIndex& index)
{
	PVRush::PVXmlTreeNodeDom* field = nullptr;

	if (index.isValid()) { // if index valid, add axis in field...
		field = nodeFromIndex(index);

		// make sure that there not already axis or regexp.
		if (!trustConfictSplitAxes(index))
			return nullptr;

		// axis adding
		if (field->typeToString() != "field") {
			QMessageBox::information((QWidget*)QObject::parent(), tr("Format builder"),
			                         tr("You must select a field first."));
			return nullptr;
		}
	} else if (!trustConfictSplitAxes(index)) {
		return nullptr;
	}

	PVRush::PVXmlTreeNodeDom* child = addAxisIn(field);
	Q_EMIT layoutChanged();

	return child;
}

/******************************************************************************
 *
 * App::PVXmlDomModel::addAxisIn
 *
 *****************************************************************************/
PVRush::PVXmlTreeNodeDom*
App::PVXmlDomModel::addAxisIn(PVRush::PVXmlTreeNodeDom* parentNode)
{
	PVRush::PVXmlTreeNodeDom* child;
	QDomElement newAxis = xmlFile.createElement("axis");
	child = new PVRush::PVXmlTreeNodeDom(newAxis);
	if (parentNode != nullptr) {
		assert(parentNode->typeToString() == "field");
		child->isOnRoot = false;
	} else {
		child->isOnRoot = true;
		parentNode = getRoot();
	}
	child->setParent(parentNode);
	parentNode->addChild(child);
	parentNode->getDom().appendChild(newAxis);

	return child;
}

/******************************************************************************
 *
 * App::PVXmlDomModel::addFilterAfter
 *
 *****************************************************************************/
PVRush::PVXmlTreeNodeDom* App::PVXmlDomModel::addFilterAfter(QModelIndex& index)
{
	PVRush::PVXmlTreeNodeDom* childPrecedent = nodeFromIndex(index); // selected node
	PVRush::PVXmlTreeNodeDom* child = nullptr;

	if (childPrecedent->typeToString() != "field") {
		QMessageBox::information((QWidget*)QObject::parent(), tr("Format builder"),
		                         tr("You must select a field first."));
	} else if (childPrecedent->typeToString() == "field") {
		// dom
		QDomElement newDom = xmlFile.createElement("filter");
		newDom.setAttribute(QString("type"), QString("regexp"));
		childPrecedent->getDom().appendChild(newDom);

		// tree
		child = new PVRush::PVXmlTreeNodeDom(newDom);
		child->setParent(childPrecedent);
		childPrecedent->addChild(child);
		Q_EMIT layoutChanged(); // refresh
	}

	return child;
}

/******************************************************************************
 *
 *  App::PVXmlDomModel::addSplitter
 *
 *****************************************************************************/
PVRush::PVXmlTreeNodeDom*
App::PVXmlDomModel::addSplitter(const QModelIndex& index,
                                        PVFilter::PVFieldsSplitterParamWidget_p splitterPlugin)
{
	assert(splitterPlugin);

	PVRush::PVXmlTreeNodeDom* child;
	PVLOG_DEBUG("App::PVXmlDomModel::addSplitter\n");
	PVRush::PVXmlTreeNodeDom* field;
	if (index.isValid()) { // add as child
		field = nodeFromIndex(index);
		if (field->typeToString() == "field") { // a splitter can be add only in field...
			if (!trustConfictSplitAxes(index)) {
				return nullptr; // we can't add more than one splitter in a field
			}
			PVLOG_DEBUG("     adding splitter in a field\n");
		} else {
			QMessageBox::information((QWidget*)QObject::parent(), tr("Format builder"),
			                         tr("You must select a field first."));
			return nullptr;
		}
	} else { // add on the root

		if (!trustConfictSplitAxes(index))
			return nullptr; // we can't add more than one splitter in a field
		field = getRoot();
	}
	PVLOG_DEBUG("     adding splitter on root node\n");
	// add node in dom
	QDomElement newDom = xmlFile.createElement(splitterPlugin->type_name());
	QString registered_name = splitterPlugin->registered_name();
	PVLOG_DEBUG("          set tag %s, type %s\n", qPrintable(splitterPlugin->type_name()),
	            qPrintable(registered_name));
	newDom.setAttribute("type", registered_name);
	field->getDom().appendChild(newDom);
	//
	// add node in tree
	child = new PVRush::PVXmlTreeNodeDom(newDom);
	child->setParent(field);
	field->addChild(child);

	// save the splitter plugin referance
	child->setSplitterPlugin(splitterPlugin);

	Q_EMIT layoutChanged();
	return child;
}

/******************************************************************************
 *
 *  App::PVXmlDomModel::addConverter
 *
 *****************************************************************************/
PVRush::PVXmlTreeNodeDom*
App::PVXmlDomModel::addConverter(const QModelIndex& index,
                                         PVFilter::PVFieldsConverterParamWidget_p converterPlugin)
{
	assert(converterPlugin);

	PVRush::PVXmlTreeNodeDom* child;
	PVLOG_DEBUG("App::PVXmlDomModel::addConverter\n");
	PVRush::PVXmlTreeNodeDom* field;
	if (index.isValid()) { // add as child
		field = nodeFromIndex(index);
		if (field->typeToString() == "field") { // a converter can be add only in field...
			PVLOG_DEBUG("     adding converter in a field\n");
		} else {
			QMessageBox::information((QWidget*)QObject::parent(), tr("Format builder"),
			                         tr("You must select a field first."));
			return nullptr;
		}
	} else { // add on the root

		field = getRoot();
	}
	PVLOG_DEBUG("     adding converter on root node\n");
	// add node in dom
	QDomElement newDom = xmlFile.createElement(converterPlugin->type_name());
	PVLOG_INFO("converterPlugin->type_name()=%s\n",
	           converterPlugin->type_name().toStdString().c_str());
	QString registered_name = converterPlugin->registered_name();
	PVLOG_DEBUG("          set tag %s, type %s\n", qPrintable(converterPlugin->type_name()),
	            qPrintable(registered_name));
	newDom.setAttribute("type", registered_name);
	field->getDom().appendChild(newDom);
	//
	// add node in tree
	child = new PVRush::PVXmlTreeNodeDom(newDom);
	child->setParent(field);
	field->addChild(child);

	// save the converter plugin referance
	child->setConverterPlugin(converterPlugin);

	Q_EMIT layoutChanged();
	return child;
}

PVRush::PVXmlTreeNodeDom* App::PVXmlDomModel::addSplitterWithAxes(
    const QModelIndex& index,
    PVFilter::PVFieldsSplitterParamWidget_p splitterPlugin,
    QStringList axesName)
{
	PVRush::PVXmlTreeNodeDom* splitter_node = addSplitter(index, splitterPlugin);

	for (auto & i : axesName) {
		// TODO: we should be able to create a field and/or an axis from separate functions !!!!!
		QDomElement newField = xmlFile.createElement(PVFORMAT_XML_TAG_FIELD_STR);
		QDomElement newAxis = xmlFile.createElement(PVFORMAT_XML_TAG_AXIS_STR);
		newAxis.setAttribute(PVFORMAT_AXIS_NAME_STR, i);
		setDefaultAttributesForAxis(newAxis);

		newField.appendChild(newAxis);
		splitter_node->getDom().appendChild(newField);
	}
	return splitter_node;
}

void App::PVXmlDomModel::setDefaultAttributesForAxis(QDomElement& elt)
{
	assert(elt.tagName() == PVFORMAT_XML_TAG_AXIS_STR);
	elt.setAttribute(PVFORMAT_AXIS_TITLECOLOR_STR, PVFORMAT_AXIS_TITLECOLOR_DEFAULT);
	elt.setAttribute(PVFORMAT_AXIS_COLOR_STR, PVFORMAT_AXIS_COLOR_DEFAULT);
	elt.setAttribute(PVFORMAT_AXIS_MAPPING_STR, PVFORMAT_AXIS_MAPPING_DEFAULT);
	elt.setAttribute(PVFORMAT_AXIS_PLOTTING_STR, PVFORMAT_AXIS_PLOTTING_DEFAULT);
	elt.setAttribute(PVFORMAT_AXIS_TYPE_STR, PVFORMAT_AXIS_TYPE_DEFAULT);
}

/******************************************************************************
 *
 * App::PVXmlDomModel::addRegExIn
 *
 *****************************************************************************/
void App::PVXmlDomModel::addRegExIn(const QModelIndex& index)
{
	// if it's a "field"
	if (index.isValid()) {
		PVRush::PVXmlTreeNodeDom* field = nodeFromIndex(index);
		if (field->typeToString() == "field") {
			if (!trustConfictSplitAxes(index))
				return;
			// dom
			QDomElement newDom = xmlFile.createElement("splitter");
			newDom.setAttribute("type", "regexp");
			field->getDom().appendChild(newDom);

			// tree
			auto child = new PVRush::PVXmlTreeNodeDom(newDom);

			child->setParent(field);
			field->addChild(child);
		} else {
			QMessageBox::information((QWidget*)QObject::parent(), tr("Format builder"),
			                         tr("You must select a field first."));
		}
	} else {
		if (!trustConfictSplitAxes(index))
			return;
		// dom
		QDomElement newDom = xmlFile.createElement("splitter");
		newDom.setAttribute("type", "regexp");
		rootNode->getDom().appendChild(newDom);
		//
		//        //tree
		auto child = new PVRush::PVXmlTreeNodeDom(newDom);
		child->setParent(getRoot());
		rootNode->addChild(child);
	}
	Q_EMIT layoutChanged();
}

/******************************************************************************
 *
 * App::PVXmlDomModel::moveDown
 *
 *****************************************************************************/
void App::PVXmlDomModel::moveDown(const QModelIndex& index)
{
	PVRush::PVXmlTreeNodeDom* child = nodeFromIndex(index);
	PVRush::PVXmlTreeNodeDom* parent = child->getParent();
	if (child->getRow() + 1 < parent->getChildren().count()) {
		PVRush::PVXmlTreeNodeDom* fllower = parent->getChild(index.row() + 1);
		// dom effect
		parent->getDom().removeChild(child->getDom());
		parent->getDom().insertAfter(child->getDom(), fllower->getDom());
		// node effect
		parent->removeChild(child);
		parent->addChildAt(child, fllower->getRow() + 1);
	}
	Q_EMIT layoutChanged();
}

/******************************************************************************
 *
 * App::PVXmlDomModel::moveUp
 *
 *****************************************************************************/
void App::PVXmlDomModel::moveUp(const QModelIndex& index)
{
	// qDebug() << "model MoveUp";

	PVRush::PVXmlTreeNodeDom* follower = nodeFromIndex(index);
	PVRush::PVXmlTreeNodeDom* parent = follower->getParent();
	if (follower->getRow() > 0) {
		PVRush::PVXmlTreeNodeDom* child = parent->getChild(index.row() - 1);
		// node in dom
		parent->getDom().removeChild(follower->getDom());
		parent->getDom().insertBefore(follower->getDom(), child->getDom());
		// node in tree
		parent->removeChild(follower);
		parent->addChildAt(follower, child->getRow());
	}
	Q_EMIT layoutChanged();
}

/******************************************************************************
 *
 * App::PVXmlDomModel::selectNext
 *
 *****************************************************************************/
QModelIndex App::PVXmlDomModel::selectNext(const QModelIndex& idx)
{
	QModelIndex newSel;
	if (idx.isValid()) {
		if (!nodeFromIndex(idx)->isOnRoot) {
			int fieldCount = nodeFromIndex(idx)->getParent()->getParent()->countChildren();
			int idFieldCurrent = nodeFromIndex(idx)->getParent()->getRow();
			int childCount = nodeFromIndex(idx)->getParent()->countChildren();
			int idChildCurrent = nodeFromIndex(idx)->getRow();
			if (idChildCurrent + 1 < childCount) { // if it isn't the last item of the field...
				newSel = index(idChildCurrent + 1, 0, idx.parent());
			} else {                                   // if it's the last item of the field...
				if (idFieldCurrent + 1 < fieldCount) { // if the node isn't on the last field...
					newSel = index(0, 0, index(idFieldCurrent + 1, 0, idx.parent().parent()));
				} else { // if the node is on the last field...
					newSel = index(0, 0, index(0, 0, idx.parent().parent()));
				}
			}
		} else {
			return newSel;
		}
	}
	return newSel; // return the index to select.
}

/******************************************************************************
 *
 * App::PVXmlDomModel::openXml
 *
 *****************************************************************************/
bool App::PVXmlDomModel::openXml(QString url)
{
	// qDebug() << "PVXmlDomModel::openXml()";
	this->urlXml = url;
	QFile fichier(this->urlXml);
	if (!fichier.exists()) {
		// the file doesn't exists
		QMessageBox qb;
		QString s;
		s.push_back("The file ");
		s.push_back(this->urlXml);
		s.push_back(" doesn't exists.");
		qb.setText(s);
		qb.exec();
		return false;
	}

	// load XML
	if (!fichier.open(QIODevice::ReadOnly)) {
		QString s;
		s.push_back("File can't be open");
		s.push_back(this->urlXml);
		QMessageBox::information((QWidget*)QObject::parent(), tr("Format builder"), s);
		return false;
	}
	QTextStream tmpTextXml(&fichier);
	tmpTextXml.setEncoding(QStringConverter::Utf8); // by default anyway
	QString err_msg;
	int err_line, err_col;
	QDomDocument doc;

	if (!doc.setContent(tmpTextXml.readAll(), false, &err_msg, &err_line, &err_col)) {
		QMessageBox msg(QMessageBox::Critical, tr("Unable to open format"),
		                tr("Unable to open format '%1'").arg(url), QMessageBox::Ok);
		msg.setInformativeText(
		    QString("XML parsing error at line %1 and column %2: ").arg(err_line).arg(err_col) +
		    err_msg);
		msg.exec();
		return false;
	}

	openXml(doc);
	return true;
}

void App::PVXmlDomModel::openXml(QDomDocument& doc)
{
	PVRush::PVFormatVersion::to_current(doc);
	xmlFile = doc;
	_original_xml_content = xmlFile.toString();
	xmlRootDom = doc.documentElement();

	// Get axes combination and remove it from the DOM
	PVRush::PVFormat format(getRootDom());
	_axes_combination.set_combination(format.get_axes_comb());
	updateAxesCombination();

	QDomElement axes_cb_elt = xmlRootDom.firstChildElement(PVFORMAT_XML_TAG_AXES_COMBINATION_STR);
	if (!axes_cb_elt.isNull()) {
		xmlRootDom.removeChild(axes_cb_elt);
	}

	rootNode = std::make_unique<PVRush::PVXmlTreeNodeDom>(PVRush::PVXmlTreeNodeDom::Type::field, "root",
	                                            xmlRootDom, xmlFile);

	beginResetModel();
	endResetModel();

	Q_EMIT layoutChanged(); // to resfresh screen
}

bool App::PVXmlDomModel::hasFormatChanged() const
{
	return xmlFile.toString() != _original_xml_content;
}

void App::PVXmlDomModel::setEltMappingPlotting(QDomElement& elt,
                                                       QString const& type,
                                                       QString const& mode_mapping,
                                                       QString const& mode_plotting)
{
	elt.setAttribute("type", type);

	QDomElement elt_map = xmlFile.createElement("mapping");
	elt_map.setAttribute("mode", mode_mapping);
	QDomElement elt_plot = xmlFile.createElement("plotting");
	elt_plot.setAttribute("mode", mode_plotting);

	elt.appendChild(elt_map);
	elt.appendChild(elt_plot);
}

/******************************************************************************
 *
 * App::PVXmlDomModel::addUrlIn
 *
 *****************************************************************************/
void App::PVXmlDomModel::addUrlIn(const QModelIndex& index)
{

	PVRush::PVXmlTreeNodeDom* child;
	PVRush::PVXmlTreeNodeDom* field;
	if (index.isValid()) { // if a item is selected...
		field = nodeFromIndex(index);
		if (field->typeToString() != "field") { // and if it's not a field
			QMessageBox::information((QWidget*)QObject::parent(), tr("Format builder"),
			                         tr("You must select a field first."));
			return;
		}
	} else {               // if no item is selected...
		field = getRoot(); // current node is the root node
	}

	// conflicts Splitter & url
	if (!trustConfictSplitAxes(index))
		return;

	// dom
	QDomElement newDom = xmlFile.createElement("splitter");
	newDom.setAttribute("type", "url");
	QDomElement f1 = xmlFile.createElement("field");
	QDomElement f2 = xmlFile.createElement("field");
	QDomElement f3 = xmlFile.createElement("field");
	QDomElement f4 = xmlFile.createElement("field");
	QDomElement f5 = xmlFile.createElement("field");
	QDomElement f6 = xmlFile.createElement("field");
	QDomElement f7 = xmlFile.createElement("field");
	QDomElement f8 = xmlFile.createElement("field");
	QDomElement f9 = xmlFile.createElement("field");
	QDomElement f10 = xmlFile.createElement("field");

	// create axis
	field->getDom().appendChild(newDom);
	QDomElement protocol = xmlFile.createElement("axis");
	QDomElement subdomain = xmlFile.createElement("axis");
	QDomElement host = xmlFile.createElement("axis");
	QDomElement domain = xmlFile.createElement("axis");
	QDomElement tld = xmlFile.createElement("axis");
	QDomElement port = xmlFile.createElement("axis");
	QDomElement url = xmlFile.createElement("axis");
	QDomElement variable = xmlFile.createElement("axis");
	QDomElement fragment = xmlFile.createElement("axis");
	QDomElement credentials = xmlFile.createElement("axis");

	// set all axis
	protocol.setAttribute("name", "Protocol");
	setEltMappingPlotting(protocol, "string", "default", "default");
	protocol.setAttribute("color", PVFORMAT_AXIS_COLOR_DEFAULT);
	protocol.setAttribute("titlecolor", PVFORMAT_AXIS_TITLECOLOR_DEFAULT);

	subdomain.setAttribute("name", "Subdomain");
	setEltMappingPlotting(subdomain, "string", "default", "default");
	subdomain.setAttribute("color", PVFORMAT_AXIS_COLOR_DEFAULT);
	subdomain.setAttribute("titlecolor", PVFORMAT_AXIS_TITLECOLOR_DEFAULT);

	host.setAttribute("name", "Host");
	setEltMappingPlotting(host, "string", "default", "default");
	host.setAttribute("color", PVFORMAT_AXIS_COLOR_DEFAULT);
	host.setAttribute("titlecolor", PVFORMAT_AXIS_TITLECOLOR_DEFAULT);

	domain.setAttribute("name", "Domain");
	setEltMappingPlotting(domain, "string", "default", "default");
	domain.setAttribute("color", PVFORMAT_AXIS_COLOR_DEFAULT);
	domain.setAttribute("titlecolor", PVFORMAT_AXIS_TITLECOLOR_DEFAULT);

	tld.setAttribute("name", "TLD");
	setEltMappingPlotting(tld, "string", "default", "default");
	tld.setAttribute("color", PVFORMAT_AXIS_COLOR_DEFAULT);
	tld.setAttribute("titlecolor", PVFORMAT_AXIS_TITLECOLOR_DEFAULT);

	port.setAttribute("name", "Port");
	setEltMappingPlotting(port, "number_uint16", "default", "port");
	port.setAttribute("color", PVFORMAT_AXIS_COLOR_DEFAULT);
	port.setAttribute("titlecolor", PVFORMAT_AXIS_TITLECOLOR_DEFAULT);

	url.setAttribute("name", "URL");
	setEltMappingPlotting(url, "string", "default", "default");
	url.setAttribute("color", PVFORMAT_AXIS_COLOR_DEFAULT);
	url.setAttribute("titlecolor", PVFORMAT_AXIS_TITLECOLOR_DEFAULT);

	variable.setAttribute("name", "Variable");
	setEltMappingPlotting(variable, "string", "default", "default");
	variable.setAttribute("color", PVFORMAT_AXIS_COLOR_DEFAULT);
	variable.setAttribute("titlecolor", PVFORMAT_AXIS_TITLECOLOR_DEFAULT);

	fragment.setAttribute("name", "Anchor");
	setEltMappingPlotting(fragment, "string", "default", "default");
	fragment.setAttribute("color", PVFORMAT_AXIS_COLOR_DEFAULT);
	fragment.setAttribute("titlecolor", PVFORMAT_AXIS_TITLECOLOR_DEFAULT);

	credentials.setAttribute("name", "Credentials");
	setEltMappingPlotting(credentials, "string", "default", "default");
	credentials.setAttribute("color", PVFORMAT_AXIS_COLOR_DEFAULT);
	credentials.setAttribute("titlecolor", PVFORMAT_AXIS_TITLECOLOR_DEFAULT);

	// add all axis
	f1.appendChild(protocol);
	f2.appendChild(subdomain);
	f3.appendChild(host);
	f4.appendChild(domain);
	f5.appendChild(tld);
	f6.appendChild(port);
	f7.appendChild(url);
	f8.appendChild(variable);
	f9.appendChild(fragment);
	f10.appendChild(credentials);

	// add all fields with axis associate
	newDom.appendChild(f1);
	newDom.appendChild(f2);
	newDom.appendChild(f3);
	newDom.appendChild(f4);
	newDom.appendChild(f5);
	newDom.appendChild(f6);
	newDom.appendChild(f7);
	newDom.appendChild(f8);
	newDom.appendChild(f9);
	newDom.appendChild(f10);

	// tree
	child = new PVRush::PVXmlTreeNodeDom(newDom);
	child->setParent(field);
	field->addChild(child);

	Q_EMIT layoutChanged();
}

/******************************************************************************
 *
 * App::PVXmlDomModel::trustConfictSplitAxes
 *
 *****************************************************************************/
bool App::PVXmlDomModel::trustConfictSplitAxes(const QModelIndex& index)
{
	PVRush::PVXmlTreeNodeDom* node;
	if (index.isValid()) {
		node = nodeFromIndex(index);
	} else {
		node = getRoot();
	}

	for (int i = 0; i < node->getChildren().count(); i++) {
		QDomElement child = node->getChildren().at(i)->getDom();
		if (child.tagName() == "axis" || child.tagName() == "RegEx" || child.tagName() == "url" ||
		    child.tagName() == "splitter") {
			QMessageBox::information((QWidget*)QObject::parent(), tr("Format builder"),
			                         tr("A field can only have one axis or one splitter. "
			                            "Delete the current node before adding a new one."));
			return false;
		}
	}

	return true;
}

void App::PVXmlDomModel::processChildrenWithField(PVCore::PVField const& field)
{
	getRoot()->getChildrenFromField(field);
	Q_EMIT layoutChanged();
}

void App::PVXmlDomModel::clearFiltersData()
{
	getRoot()->clearFiltersData();
}

void App::PVXmlDomModel::updateFiltersDataDisplay()
{
	getRoot()->updateFiltersDataDisplay();
}

void App::PVXmlDomModel::updateFieldsLinearId()
{
	getRoot()->updateFieldLinearId((PVCol)0);
}

void App::PVXmlDomModel::setAxesNames(QStringList const& names)
{
	getRoot()->setAxesNames(names, (PVCol)0);

	Q_EMIT layoutChanged();
}

void App::PVXmlDomModel::updateAxesCombination()
{
	bool was_default = _axes_combination.is_default();

	try {
		PVRush::PVFormat format(getRootDom());
		_axes = format.get_axes();
	} catch (PVRush::PVFormatInvalid const&) {
		// The format is empty, keep axes combination empty too.
		_axes.clear();
		_axes_combination.reset_to_default();
		return;
	}

	if (was_default) {
		_axes_combination.reset_to_default();
	} else {
		auto comb = _axes_combination.get_combination();
		std::erase_if(comb, [this](PVCol c) { return c >= _axes.size(); });
		if (comb.empty()) {
			comb.resize(_axes.size());
			std::iota(comb.begin(), comb.end(), PVCol(0));
		}
		_axes_combination.set_combination(comb);
	}
}

size_t App::PVXmlDomModel::get_axes_count() const
{
	return getRootDom().elementsByTagName("axis").length();
}

void App::PVXmlDomModel::set_python_script(const QString& python_script, bool is_path, bool disabled)
{
	QDomDocument doc = getRootDom().toDocument();

	const QString& python_script_tag = "python-script";
	QDomElement python_script_element = getRootDom().firstChildElement(python_script_tag);

	QDomElement newNodeTag = doc.createElement(python_script_tag); 
	newNodeTag.setAttribute("path", is_path ? "1" : "0");
	newNodeTag.setAttribute("disabled", disabled ? "1" : "0");
	
	if (python_script_element.isNull()) {
		getRootDom().appendChild(newNodeTag);
	}
	else {
		getRootDom().replaceChild(newNodeTag, python_script_element);
	}

	QString text_node_content = python_script;
	if (not is_path) {
		text_node_content = PVCore::serialize_base64<QString>(text_node_content);
	}
	QDomText newNodeText = doc.createTextNode(text_node_content);
	newNodeTag.appendChild(newNodeText);
}

QString App::PVXmlDomModel::get_python_script(bool& is_path, bool& disabled) const
{
	QDomElement python_script_element = getRootDom().firstChildElement("python-script");

	QString python_script = python_script_element.text();
	is_path = python_script_element.attribute("path", "1").toUInt();
	disabled = python_script_element.attribute("disabled", "1").toUInt();
	if (not is_path) {
		python_script = PVCore::deserialize_base64<QString>(python_script);
	}
	return python_script;
}
