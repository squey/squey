///! \file PVXmlTreeView.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011


#include <PVXmlTreeView.h>
#include <PVXmlDomModel.h>
#include<QMouseEvent>
#include<QTreeView>
#include<QTreeWidgetItem>
#include<QRectF>
#include <QAbstractItemView>
#include <QStyleOptionViewItem>
#include <QItemSelectionModel>
#include <QAbstractItemModel>
#include<QRect>
#include<QScrollBar>
#include<qglobal.h>
#include<QDragMoveEvent>

#define dbg()  {qDebug()<<__FILE__<<__LINE__;}



/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::PVXmlTreeView
 *
 *****************************************************************************/
PVInspector::PVXmlTreeView::PVXmlTreeView(QWidget * parent ) :QTreeView(parent) //
{
    setDragEnabled(TRUE);
    setAcceptDrops(TRUE);
    setDropIndicatorShown(TRUE);
    isDraging = FALSE;
    isEditing = false;
    setObjectName("PVXmlTreeView");
}




/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::~PVXmlTreeView
 *
 *****************************************************************************/
PVInspector::PVXmlTreeView::~PVXmlTreeView() {
}




/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::addAxisIn
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeView::addAxisIn() {
  addNode(addAxis);
}



/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::addFilterAfter
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeView::addFilterAfter() {
  addNode(addFilter);
}



/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::addRegExIn
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeView::addRegExIn() {
  addNode(addRegEx);
}



/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::addUrlIn
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeView::addUrlIn() {
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
void PVInspector::PVXmlTreeView::addNode(AddType type){

	QModelIndex index;
	QModelIndex indexToSelect;
	
	int numberOfSelectedIndexes = selectedIndexes().count();//get the number of selected indexes.
	
	if (numberOfSelectedIndexes>0) {
		index = selectedIndexes().at(0);//get the selected index.
	}
	
	switch (type) {
		//if we want to add a regexp
		case addRegEx:
			getModel()->addRegExIn(index);
			indexToSelect = index.child(getModel()->nodeFromIndex(index)->countChildren() - 1, 0);//last child of the node.
			break;

		//if we want to add a filter
		case addFilter:
			getModel()->addFilterAfter(index);
			indexToSelect = index.child(0, 0);//the first child
			break;

		//if we want to add an axis
		case addAxis:
			getModel()->addAxisIn(index);//add the element
			if ( numberOfSelectedIndexes > 0) {
				indexToSelect = index.child(0, 0).child(0,0);//set the index to select after the adding.
			}
			break;

		//if we want to add a splitter url
		case addUrl:
			getModel()->addUrlIn(index);//add the element
			indexToSelect = index.child(getModel()->nodeFromIndex(index)->countChildren() - 1, 0);//last child of the node.
			expandAll();
		break;
		
		default:
			break;
	}
	
	if (index.isValid()) {
		expandRecursive(index);
	}
	
	//selection of the item just after the creation
	if (indexToSelect.isValid()) {
		selectionModel()->select(indexToSelect, QItemSelectionModel::ClearAndSelect);
		emit clicked(indexToSelect);//the new item become the new selected item.
	}
}




/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::applyModification
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeView::applyModification(PVXmlParamWidget *paramBord,QModelIndex& ){
  //if(selectedIndexes())PVLOG_ERROR("selectedIndexes() is null in PVInspector::PVXmlTreeView::applyModification(PVXmlParamWidget *paramBord)\n");
	if (paramBord==NULL) {
		PVLOG_ERROR("paramBord is null in PVInspector::PVXmlTreeView::applyModification(PVXmlParamWidget *paramBord)\n");
	}
	
	if (getModel()==NULL) {
		PVLOG_ERROR("getModel() is null in PVInspector::PVXmlTreeView::applyModification(PVXmlParamWidget *paramBord)\n");
	}
  

	if (selectedIndexes().count()>0) {
		//if an item is selected
		QModelIndex index = selectedIndexes().at(0);
		if (!index.isValid()) {
			PVLOG_ERROR("index invalid in PVInspector::PVXmlTreeView::applyModification(PVXmlParamWidget *paramBord)\n");
		}
		getModel()->applyModification(index,paramBord); 
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
void PVInspector::PVXmlTreeView::deleteSelection(){
    //deleting
    for(int i=1;i<selectedIndexes().size();i++){
        QModelIndex index = selectedIndexes().at(i-1);
        QModelIndex parentIndex = index.parent();
        getModel()->deleteSelection(index);
        if (parentIndex.isValid()) {
            collapse(parentIndex);
            expandRecursive(parentIndex);
            if(parentIndex.isValid())emit clicked(parentIndex);
	    else PVLOG_WARN("Can't select parentIndex in PVInspector::PVXmlTreeView::deleteSelection()\n");
        }else{
        }
    }
    
}




/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::expandRecursive
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeView::expandRecursive(const QModelIndex &index){
    expand(index);

    //expand for all child
    for(int i=0;i<getModel()->rowCount(index);i++) {
        expandRecursive(index.child(i, 0));
    }

    //update size of the first column
    //calculate column size needed.
    if (index.isValid()) {
        int l_width = itemDelegate(index)->sizeHint(QStyleOptionViewItem(), index).width();
        int l_offset = getModel()->countParent(index)*20;

        //resize column
        if (columnWidth(0) < l_width + l_offset+30) {
            setColumnWidth(0, l_width + l_offset+30);
        }
    }
}




/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::getModel
 *
 *****************************************************************************/
PVInspector::PVXmlDomModel* PVInspector::PVXmlTreeView::getModel(){
  if(model()==NULL){
    PVLOG_ERROR("no model in PVInspector::PVXmlTreeView::getModel()\n");
    return NULL;
  }
  return ((PVXmlDomModel*) model());
}



/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::mouseDoubleClickEvent
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeView::mouseDoubleClickEvent ( QMouseEvent * event ){
        QTreeView::mouseDoubleClickEvent(event);
        isEditing = true;
}



/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::mousePressEvent
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeView::mousePressEvent(QMouseEvent * event) {
        for (int i = 0; i < selectedIndexes().count(); i++) {
                if (selectedIndexes().at(i).isValid()){
                        selectionModel()->select(selectedIndexes().at(i), QItemSelectionModel::Clear); //valid index...
                }
        }

        QTreeView::mousePressEvent(event);
        //QTreeView::
        QModelIndex index;
        if (index.isValid()) {
                PVLOG_DEBUG("emit clicked() on an invalid index in PVInspector::PVXmlTreeView::mousePressEvent()\n");
        }
        emit clicked(index); //keep it anyway, if the index is not valid, it update toolsbar enabled tools.
}



/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::moveDown
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeView::moveDown() {
    if (selectedIndexes().count() > 0) {//if an item is selected
        QModelIndex index = selectedIndexes().at(0);
        getModel()->moveDown(index);
        if (index.row() < getModel()->nodeFromIndex(index)->getParent()->countChildren()-1)//if child isn't the last
            if(index.parent().child(index.row() + 1, 0).isValid()){//valid index...
	      selectionModel()->select(index.parent().child(index.row() + 1, 0), QItemSelectionModel::ClearAndSelect);
	      emit clicked(index.parent().child(index.row() + 1, 0));
	    }
    }
}



/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::moveUp
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeView::moveUp() {
    if (selectedIndexes().count() > 0) {//if an item is selected
        QModelIndex index = selectedIndexes().at(0);
        getModel()->moveUp(index);
        if (index.row() > 0){//if child isn't the first
	  if(index.parent().child(index.row() - 1, 0).isValid()){//valid index...
            selectionModel()->select(index.parent().child(index.row() - 1, 0), QItemSelectionModel::ClearAndSelect);
	    emit clicked(index.parent().child(index.row() - 1, 0));
	  }
	}
    }
}

/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::slotDataHasChanged
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeView::slotDataHasChanged(const QModelIndex & index, const QModelIndex & ){
        PVLOG_DEBUG("PVInspector::PVXmlTreeView::slotDataHasChanged\n");
        emit clicked(index);
}

/******************************************************************************
 *
 * PVInspector::PVXmlTreeView::slotSelectNext
 *
 *****************************************************************************/
void PVInspector::PVXmlTreeView::slotSelectNext(){
  if (selectedIndexes().count() > 0) {//if an item is selected...
    QModelIndex index = selectedIndexes().at(0);
    if(index.isValid()){//if the index is valid...
      if(index.parent().isValid()){
	QModelIndex selectIndex =getModel()->selectNext(index);
	if(selectIndex.isValid()){//if the new index is valid
	    
            if(isEditing){
                    PVLOG_DEBUG("has EditFocus\n");
                    isEditing = false;
                    selectionModel()->select(index, QItemSelectionModel::ClearAndSelect);
                    emit clicked(index);
            }else{
                    PVLOG_DEBUG("hasn't EditFocus\n");
                    selectionModel()->select(selectIndex, QItemSelectionModel::ClearAndSelect);
                    emit clicked(selectIndex);
            }
	}
      }
    }
  }
}

