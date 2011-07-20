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
//#include <pvcore/PVXmlTreeNodeDom.h>
#include <PVXmlParamWidget.h>
#include <PVXmlDomModel.h>
#include <pvfilter/PVFieldsFilterParamWidget.h>


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
     * add a new splitter in DOM refering to the splitter plugin
     * @param splitterPlugin : new instance of the plugin requesting the new splitter
     */
    void addSplitter(PVFilter::PVFieldsSplitterParamWidget_p splitterPlugin);
    
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

    QModelIndex getSelectedIndex() {
        QModelIndex index;

        int numberOfSelectedIndexes = selectedIndexes().count(); //get the number of selected indexes.

        if (numberOfSelectedIndexes > 0) {
            index = selectedIndexes().at(0); //get the selected index.
        }
        return index;
    }
      
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

