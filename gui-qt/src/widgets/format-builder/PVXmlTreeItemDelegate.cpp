///! \file PVXmlTreeItemDelegate.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#include <PVXmlTreeItemDelegate.h>




/******************************************************************************
 *
 * PVInspector::PVXmlTreeItemDelegate::PVXmlTreeItemDelegate
 *
 *****************************************************************************/
PVInspector::PVXmlTreeItemDelegate::PVXmlTreeItemDelegate() :
        QAbstractItemDelegate()
{
    
}





/******************************************************************************
 *
 * PVInspector::PVXmlTreeItemDelegate::~PVXmlTreeItemDelegate
 *
 *****************************************************************************/
PVInspector::PVXmlTreeItemDelegate::~PVXmlTreeItemDelegate() {
}

//void MyItemDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const {
//    
//    if(true){
//        
//    }
//    painter->drawRect(option.rect.topLeft().x(), option.rect.topLeft().y(), 60, 25);
//    //painter->drawRect(option.rect);
//    //painter->drawRect(5, 5, 60, 250);
//    
//    QString text = ((NodeDom*)(index.internalPointer()))->getName();
//    painter->drawText(option.rect.topLeft().x()+3, option.rect.topLeft().y()+17, text);
//}



/******************************************************************************
 *
 *  PVInspector::PVXmlTreeItemDelegate::sizeHint
 *
 *****************************************************************************/
QSize PVInspector::PVXmlTreeItemDelegate::sizeHint(const QStyleOptionViewItem &option, const QModelIndex &index) const {
    
    return QSize(300, 30);
}


    