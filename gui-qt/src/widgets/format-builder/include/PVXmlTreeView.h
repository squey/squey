//! \file PVXmlTreeView.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef MYTREEVIEW_H
#define	MYTREEVIEW_H


#include<QTreeView>
#include<QTreeWidgetItem>

#include <QMouseEvent>


#include <iostream>
#include <PVXmlTreeNodeDom.h>
#include <PVXmlParamWidget.h>
#include <PVXmlDomModel.h>



//QTreeView

const int ExtraHeight = 3;

namespace PVInspector {
  

class PVXmlTreeView : public QTreeView /* public QAbstractItemView*/{
    Q_OBJECT

public:
  enum AddType {
        addRegEx, addFilter, addAxis, addUrl
    };
    PVXmlTreeView(QWidget * parent = NULL);
    virtual ~PVXmlTreeView();
    

    void addAxisIn();
    
    /**
     * Add a new Filter after the selected element.
     */
    void addFilterAfter();
    
    /**
     * Add a new RegEx after the selected element.
     */
    void addRegExIn();
    void addUrlIn();
    
    
    void addNode(AddType type);
    
    void applyModification(PVXmlParamWidget *paramBord,QModelIndex& index);
  
    

    /**
     * Delete the selected Item
     */
      void deleteSelection();
      
    /**
     * expand recursively fields.
     * @param index
     */
      void expandRecursive(const QModelIndex &index);
      
      virtual void mouseDoubleClickEvent ( QMouseEvent * event );
      virtual void mousePressEvent ( QMouseEvent * event );
         
      PVXmlDomModel * getModel();
      
    /**
     * Move down the selected element.
     */
    void moveDown();
    
    /**
     * Move up the selected element.
     */
    void moveUp();
    
    


    

protected:

private:
    bool isDraging;
    bool isEditing;
public slots:
    void slotDataHasChanged(const QModelIndex & , const QModelIndex & );
    void slotSelectNext();
    signals:
    void refresh();
};
}
#endif	/* MYTREEVIEW_H */

