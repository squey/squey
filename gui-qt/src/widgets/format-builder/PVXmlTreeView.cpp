/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <PVXmlTreeView.h>
#include <PVXmlDomModel.h>
#include <QMouseEvent>
#include <QTreeView>
#include <QTreeWidgetItem>
#include <QRectF>
#include <QAbstractItemView>
#include <QStyleOptionViewItem>
#include <QItemSelectionModel>
#include <QAbstractItemModel>
#include <QString>
#include <QRect>
#include <QScrollBar>
#include <qglobal.h>
#include <QDragMoveEvent>

#define dbg()                                                                                      \
	{                                                                                              \
		qDebug() << __FILE__ << __LINE__;                                                          \
	}

/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::PVXmlTreeView
 *
 *****************************************************************************/
PVInspector::PVXmlTreeView::PVXmlTreeView(QWidget* parent) : QTreeView(parent) //
{
	setDragEnabled(true);
	setAcceptDrops(true);
	setDropIndicatorShown(true);
	isDraging = false;
	isEditing = false;
	setObjectName("PVXmlTreeView");
}

/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::~PVXmlTreeView
 *
 *****************************************************************************/
PVInspector::PVXmlTreeView::~PVXmlTreeView() = default;

/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::addAxisIn
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeView::addAxisIn()
{
	PVLOG_DEBUG("PVInspector::PVXmlTreeView::addAxisIn\n");

	QModelIndex index = getInsertionIndex();
	auto* node = getModel()->addAxisIn(index);

	postInsertion(index, node);
}

/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::addFilterAfter
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeView::addFilterAfter()
{
	PVLOG_DEBUG("PVInspector::PVXmlTreeView::addFilterAfter\n");

	QModelIndex index = getInsertionIndex();
	auto* node = getModel()->addFilterAfter(index);

	postInsertion(index, node);
}

/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::addSplitter
 *
 *****************************************************************************/
PVRush::PVXmlTreeNodeDom*
PVInspector::PVXmlTreeView::addSplitter(PVFilter::PVFieldsSplitterParamWidget_p splitterPlugin)
{
	PVLOG_DEBUG("PVInspector::PVXmlTreeView::addSplitter\n");

	QModelIndex index = getInsertionIndex();
	auto* node = getModel()->addSplitter(index, splitterPlugin);

	postInsertion(index, node);

	return node;
}

/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::addConverter
 *
 *****************************************************************************/
PVRush::PVXmlTreeNodeDom*
PVInspector::PVXmlTreeView::addConverter(PVFilter::PVFieldsConverterParamWidget_p converterPlugin)
{
	PVLOG_DEBUG("PVInspector::PVXmlTreeView::addConverter \n");

	QModelIndex index = getInsertionIndex();
	auto* node = getModel()->addConverter(index, converterPlugin);

	postInsertion(index, node);

	return node;
}

/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::addRegExIn
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeView::addRegExIn()
{
	addNode(addRegEx);
}

/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::addUrlIn
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeView::addUrlIn()
{
	addNode(addUrl);
}

/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::addNode
 *
 *****************************************************************************/
/**
* add a new node (axis, filter, regexp or url)
* @param type
*/
void PVInspector::PVXmlTreeView::addNode(AddType type)
{

	QModelIndex index = getInsertionIndex();
	QModelIndex indexToSelect;

	switch (type) {
	// if we want to add a regexp
	case addRegEx:
		getModel()->addRegExIn(index);
		indexToSelect = index.child(getModel()->nodeFromIndex(index)->countChildren() - 1,
		                            0); // last child of the node.
		break;

	// if we want to add a splitter url
	case addUrl:
		getModel()->addUrlIn(index); // add the element
		indexToSelect = index.child(getModel()->nodeFromIndex(index)->countChildren() - 1,
		                            0); // last child of the node.
		expandAll();
		break;

	default:
		break;
	}

	if (index.isValid()) {
		expandRecursive(index);
	}

	// selection of the item just after the creation
	if (indexToSelect.isValid()) {
		selectionModel()->select(indexToSelect, QItemSelectionModel::ClearAndSelect);
		Q_EMIT clicked(indexToSelect); // the new item become the new selected item.
	}
}

/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::applyModification
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeView::applyModification(PVXmlParamWidget* paramBord, QModelIndex&)
{
	// if(selectedIndexes())PVLOG_ERROR("selectedIndexes() is null in
	// PVInspector::PVXmlTreeView::applyModification(PVXmlParamWidget *paramBord)\n");
	if (paramBord == nullptr) {
		PVLOG_ERROR("paramBord is null in "
		            "PVInspector::PVXmlTreeView::applyModification(PVXmlParamWidget *paramBord)\n");
	}

	if (getModel() == nullptr) {
		PVLOG_ERROR("getModel() is null in "
		            "PVInspector::PVXmlTreeView::applyModification(PVXmlParamWidget *paramBord)\n");
	}

	if (selectedIndexes().count() > 0) {
		// if an item is selected
		QModelIndex index = selectedIndexes().at(0);
		if (!index.isValid()) {
			PVLOG_ERROR("index invalid in "
			            "PVInspector::PVXmlTreeView::applyModification(PVXmlParamWidget "
			            "*paramBord)\n");
		}
		getModel()->applyModification(index, paramBord);
		if (index.isValid()) {
			expandRecursive(index);
		}
	}
}

/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::deleteSelection
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeView::deleteSelection()
{
	QModelIndexList sels_idx = selectedIndexes();
	QModelIndex parent;

	for (QModelIndex const& index : sels_idx) {
		if (index.column() != 0) {
			continue;
		}
		QModelIndex parentIndex = index.parent();
		if (parentIndex.isValid()) {
			collapse(parentIndex);
			expandRecursive(parentIndex);
			if (parentIndex.isValid()) {
				Q_EMIT clicked(parentIndex);
			}
		}
		getModel()->deleteSelection(index);
		parent = index.parent();
	}

	if (!parent.isValid()) {
		// i there is no parent to select, we try a sibling
		parent = model()->index(0, 0, parent);
	}

	setCurrentIndex(parent);
}

/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::expandRecursive
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeView::expandRecursive(const QModelIndex& index)
{
	expand(index);

	// expand for all child
	for (int i = 0; i < getModel()->rowCount(index); i++) {
		expandRecursive(index.child(i, 0));
	}

	// update size of the first column
	// calculate column size needed.
	if (index.isValid()) {
		int l_width = itemDelegate(index)->sizeHint(QStyleOptionViewItem(), index).width();
		int l_offset = getModel()->countParent(index) * 20;

		// resize column
		if (columnWidth(0) < l_width + l_offset + 30) {
			setColumnWidth(0, l_width + l_offset + 30);
		}
	}
}

/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::getModel
 *
 *****************************************************************************/
PVInspector::PVXmlDomModel* PVInspector::PVXmlTreeView::getModel()
{
	if (model() == nullptr) {
		PVLOG_ERROR("no model in PVInspector::PVXmlTreeView::getModel()\n");
		return nullptr;
	}
	return ((PVXmlDomModel*)model());
}

/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::mouseDoubleClickEvent
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeView::mouseDoubleClickEvent(QMouseEvent* event)
{
	QTreeView::mouseDoubleClickEvent(event);
	isEditing = true;
}

/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::moveDown
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeView::moveDown()
{
	QModelIndex index = currentIndex();

	if (!index.isValid()) {
		return;
	}

	QModelIndex new_index = index.sibling(index.row() + 1, index.column());

	if (!new_index.isValid()) {
		return;
	}

	getModel()->moveDown(index);
	setCurrentIndex(new_index);

	Q_EMIT clicked(new_index);
}

/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::moveUp
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeView::moveUp()
{
	QModelIndex index = currentIndex();

	if (!index.isValid()) {
		return;
	}

	QModelIndex new_index = index.sibling(index.row() - 1, index.column());

	if (!new_index.isValid()) {
		return;
	}

	getModel()->moveUp(index);
	setCurrentIndex(new_index);

	Q_EMIT clicked(new_index);
}

/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::slotDataHasChanged
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeView::slotDataHasChanged(const QModelIndex& index,
                                                    const QModelIndex&,
                                                    const QVector<int>&)
{
	PVLOG_DEBUG("PVInspector::PVXmlTreeView::slotDataHasChanged\n");
	Q_EMIT clicked(index);
}

/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::slotSelectNext
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeView::slotSelectNext()
{
	if (selectedIndexes().count() > 0) { // if an item is selected...
		QModelIndex index = selectedIndexes().at(0);
		if (index.isValid()) { // if the index is valid...
			if (index.parent().isValid()) {
				QModelIndex selectIndex = getModel()->selectNext(index);
				if (selectIndex.isValid()) { // if the new index is valid

					if (isEditing) {
						PVLOG_DEBUG("has EditFocus\n");
						isEditing = false;
						selectionModel()->select(index, QItemSelectionModel::ClearAndSelect);
						Q_EMIT clicked(index);
					} else {
						PVLOG_DEBUG("hasn't EditFocus\n");
						selectionModel()->select(selectIndex, QItemSelectionModel::ClearAndSelect);
						Q_EMIT clicked(selectIndex);
					}
				}
			}
		}
	}
}
