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
 * App::PVXmlTreeView::PVXmlTreeView
 *
 *****************************************************************************/
App::PVXmlTreeView::PVXmlTreeView(QWidget* parent) : QTreeView(parent) //
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
 * App::PVXmlTreeView::~PVXmlTreeView
 *
 *****************************************************************************/
App::PVXmlTreeView::~PVXmlTreeView() = default;

/******************************************************************************
 *
 * App::PVXmlTreeView::addAxisIn
 *
 *****************************************************************************/
void App::PVXmlTreeView::addAxisIn()
{
	PVLOG_DEBUG("App::PVXmlTreeView::addAxisIn\n");

	QModelIndex index = getInsertionIndex();
	auto* node = getModel()->addAxisIn(index);

	postInsertion(index, node);
}

/******************************************************************************
 *
 * App::PVXmlTreeView::addFilterAfter
 *
 *****************************************************************************/
void App::PVXmlTreeView::addFilterAfter()
{
	PVLOG_DEBUG("App::PVXmlTreeView::addFilterAfter\n");

	QModelIndex index = getInsertionIndex();
	auto* node = getModel()->addFilterAfter(index);

	postInsertion(index, node);
}

/******************************************************************************
 *
 * App::PVXmlTreeView::addSplitter
 *
 *****************************************************************************/
PVRush::PVXmlTreeNodeDom*
App::PVXmlTreeView::addSplitter(PVFilter::PVFieldsSplitterParamWidget_p splitterPlugin)
{
	PVLOG_DEBUG("App::PVXmlTreeView::addSplitter\n");

	QModelIndex index = getInsertionIndex();
	auto* node = getModel()->addSplitter(index, splitterPlugin);

	postInsertion(index, node);

	return node;
}

/******************************************************************************
 *
 * App::PVXmlTreeView::addConverter
 *
 *****************************************************************************/
PVRush::PVXmlTreeNodeDom*
App::PVXmlTreeView::addConverter(PVFilter::PVFieldsConverterParamWidget_p converterPlugin)
{
	PVLOG_DEBUG("App::PVXmlTreeView::addConverter \n");

	QModelIndex index = getInsertionIndex();
	auto* node = getModel()->addConverter(index, converterPlugin);

	postInsertion(index, node);

	return node;
}

/******************************************************************************
 *
 * App::PVXmlTreeView::addRegExIn
 *
 *****************************************************************************/
void App::PVXmlTreeView::addRegExIn()
{
	addNode(addRegEx);
}

/******************************************************************************
 *
 * App::PVXmlTreeView::addUrlIn
 *
 *****************************************************************************/
void App::PVXmlTreeView::addUrlIn()
{
	addNode(addUrl);
}

/******************************************************************************
 *
 * App::PVXmlTreeView::addNode
 *
 *****************************************************************************/
/**
* add a new node (axis, filter, regexp or url)
* @param type
*/
void App::PVXmlTreeView::addNode(AddType type)
{

	QModelIndex idx = getInsertionIndex();
	QModelIndex indexToSelect;

	switch (type) {
	// if we want to add a regexp
	case addRegEx:
		getModel()->addRegExIn(idx);
		indexToSelect = getModel()->index(getModel()->nodeFromIndex(idx)->countChildren() - 1, 0, idx); // last child of the node.
		break;

	// if we want to add a splitter url
	case addUrl:
		getModel()->addUrlIn(idx); // add the element
		indexToSelect = getModel()->index(getModel()->nodeFromIndex(idx)->countChildren() - 1, 0, idx); // last child of the node.
		expandAll();
		break;

	default:
		break;
	}

	if (idx.isValid()) {
		expandRecursive(idx);
	}

	// selection of the item just after the creation
	if (indexToSelect.isValid()) {
		selectionModel()->select(indexToSelect, QItemSelectionModel::ClearAndSelect);
		Q_EMIT clicked(indexToSelect); // the new item become the new selected item.
	}
}

/******************************************************************************
 *
 * App::PVXmlTreeView::applyModification
 *
 *****************************************************************************/
void App::PVXmlTreeView::applyModification(PVXmlParamWidget* paramBord, QModelIndex&)
{
	// if(selectedIndexes())PVLOG_ERROR("selectedIndexes() is null in
	// App::PVXmlTreeView::applyModification(PVXmlParamWidget *paramBord)\n");
	if (paramBord == nullptr) {
		PVLOG_ERROR("paramBord is null in "
		            "App::PVXmlTreeView::applyModification(PVXmlParamWidget *paramBord)\n");
	}

	if (getModel() == nullptr) {
		PVLOG_ERROR("getModel() is null in "
		            "App::PVXmlTreeView::applyModification(PVXmlParamWidget *paramBord)\n");
	}

	if (selectedIndexes().count() > 0) {
		// if an item is selected
		QModelIndex index = selectedIndexes().at(0);
		if (!index.isValid()) {
			PVLOG_ERROR("index invalid in "
			            "App::PVXmlTreeView::applyModification(PVXmlParamWidget "
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
 * App::PVXmlTreeView::deleteSelection
 *
 *****************************************************************************/
void App::PVXmlTreeView::deleteSelection()
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
 * App::PVXmlTreeView::expandRecursive
 *
 *****************************************************************************/
void App::PVXmlTreeView::expandRecursive(const QModelIndex& index)
{
	expand(index);

	// expand for all child
	for (int i = 0; i < getModel()->rowCount(index); i++) {
		expandRecursive(getModel()->index(i, 0, index));
	}

	// update size of the first column
	// calculate column size needed.
	if (index.isValid()) {
		int l_width = itemDelegateForIndex(index)->sizeHint(QStyleOptionViewItem(), index).width();
		int l_offset = getModel()->countParent(index) * 20;

		// resize column
		if (columnWidth(0) < l_width + l_offset + 30) {
			setColumnWidth(0, l_width + l_offset + 30);
		}
	}
}

/******************************************************************************
 *
 * App::PVXmlTreeView::getModel
 *
 *****************************************************************************/
App::PVXmlDomModel* App::PVXmlTreeView::getModel()
{
	if (model() == nullptr) {
		PVLOG_ERROR("no model in App::PVXmlTreeView::getModel()\n");
		return nullptr;
	}
	return ((PVXmlDomModel*)model());
}

/******************************************************************************
 *
 * App::PVXmlTreeView::mouseDoubleClickEvent
 *
 *****************************************************************************/
void App::PVXmlTreeView::mouseDoubleClickEvent(QMouseEvent* event)
{
	QTreeView::mouseDoubleClickEvent(event);
	isEditing = true;
}

/******************************************************************************
 *
 * App::PVXmlTreeView::moveDown
 *
 *****************************************************************************/
void App::PVXmlTreeView::moveDown()
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
 * App::PVXmlTreeView::moveUp
 *
 *****************************************************************************/
void App::PVXmlTreeView::moveUp()
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
 * App::PVXmlTreeView::slotDataHasChanged
 *
 *****************************************************************************/
void App::PVXmlTreeView::slotDataHasChanged(const QModelIndex& index,
                                                    const QModelIndex&,
                                                    const QVector<int>&)
{
	PVLOG_DEBUG("App::PVXmlTreeView::slotDataHasChanged\n");
	Q_EMIT clicked(index);
}

/******************************************************************************
 *
 * App::PVXmlTreeView::slotSelectNext
 *
 *****************************************************************************/
void App::PVXmlTreeView::slotSelectNext()
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
