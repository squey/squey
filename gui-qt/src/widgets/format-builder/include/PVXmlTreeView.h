/**
 * \file PVXmlTreeView.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef MYTREEVIEW_H
#define	MYTREEVIEW_H


#include<QTreeView>
#include<QTreeWidgetItem>

#include <QMouseEvent>


#include <iostream>
#include <PVXmlParamWidget.h>
#include <PVXmlDomModel.h>
#include <pvkernel/filter/PVFieldsFilterParamWidget.h>


//QTreeView

const int ExtraHeight = 3;

namespace PVRush {
class PVXmlTreeNodeDom;
}

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
	PVRush::PVXmlTreeNodeDom* addSplitter(PVFilter::PVFieldsSplitterParamWidget_p splitterPlugin);
	PVRush::PVXmlTreeNodeDom* addConverter(PVFilter::PVFieldsConverterParamWidget_p converterPlugin);
	
	PVRush::PVXmlTreeNodeDom* processChildrenWithField();
    
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
		  QModelIndexList lsel = selectedIndexes();
		  int numberOfSelectedIndexes = lsel.count();
		  for (int i = 0; i < numberOfSelectedIndexes; i++) {
			  QModelIndex tmp = lsel[i];
			  PVLOG_DEBUG("selected index: %d %d\n", tmp.row(), tmp.column());
		  }

		  return (numberOfSelectedIndexes > 0) ? lsel[0] : QModelIndex();
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

