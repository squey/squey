/**
 * \file PVXmlTreeItemDelegate.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef MYITEMDELEGATE_H
#define	MYITEMDELEGATE_H
#include<QAbstractItemDelegate>
#include <QPainter>
#include <QSize>
 #include <QTextEdit>


namespace PVInspector{
class PVXmlTreeItemDelegate: public QAbstractItemDelegate{
public:
    PVXmlTreeItemDelegate();
    virtual ~PVXmlTreeItemDelegate();
    
    
    //
    //virtual void paint(QPainter *painter, const QStyleOptionViewItem &option,const QModelIndex &index) const ;
    
    /**
     * Define the box size of a widget like item.
     * @param option
     * @param index
     * @return 
     */
    virtual QSize sizeHint(const QStyleOptionViewItem &option,const QModelIndex &index) const ;
    
    
    
private:

};
}
#endif	/* MYITEMDELEGATE_H */

