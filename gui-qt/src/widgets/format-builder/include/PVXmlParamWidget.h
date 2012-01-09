//! \file PVXmlParamWidget.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVXMLPARAMWIDGET_H
#define	PVXMLPARAMWIDGET_H
#include<QWidget>
 #include <QVBoxLayout>
#include<QVariant>
#include<QLabel>
#include<QModelIndex>
#include <QPushButton>
#include <QDebug>
#include <QPushButton>
#include <QMessageBox>
#include <iostream>
#include <QComboBox>
#include <QDir>
#include <QStringList>
#include <QString>


#include <pvkernel/rush/PVXmlTreeNodeDom.h>
#include <PVXmlParamTextEdit.h>
#include <picviz/plugins.h>
#include <PVXmlParamColorDialog.h>
#include <PVXmlParamWidgetBoardAxis.h>
#include <PVXmlParamWidgetBoardFilter.h>
#include <PVXmlParamWidgetBoardSplitterRegEx.h>
//#include <PVXmlDomModel.h>

//#include "include/NodeDom.h"

namespace PVInspector{
class PVFormatBuilderWidget;

class PVXmlParamWidget: public QWidget {
    Q_OBJECT
   
public:
     enum Type {
        filterParam, splitterParam, no
    };
    
    PVXmlParamWidget(PVFormatBuilderWidget* parent);

    virtual ~PVXmlParamWidget();

    /**
     * draw the filter parameter box.
     * @param nodeFilter
     */
    void drawForFilter(PVRush::PVXmlTreeNodeDom  *nodeFilter);
    /**
     * clear the parameter box.
     * @param index
     * @todo will be delete when all splitter will be plugins
     */
    void drawForNo(QModelIndex index);
    /**
     * Draw the regexp parameter box.
     * @param nodeSplitter
     * @todo will be delete when all splitter will be plugins
     */
    void drawForRegEx(PVRush::PVXmlTreeNodeDom  *nodeSplitter);
    /**
     * Draw the regexp parameter box.
     * @param nodeSplitter
     */
    void drawForSplitter(PVRush::PVXmlTreeNodeDom  *nodeSplitter);
    /**
     * Dras the axis parameter box.
     * @param nodeOnClick
     */
    void drawForAxis(PVRush::PVXmlTreeNodeDom  *nodeOnClick);
    
    /**
     * this is used to update parameter board
     */
    void addListWidget();
    /**
     * this is used to update parameter board
     */
    void removeListWidget();
    
    /**
     * get the parameter, form his index
     * @param i
     * @return 
     */
    QVariant getParam(int i);
    
    /**
     * get the parameter, from his name.
     * @param nameParam
     * @return 
     */
    QVariant getParamVariantByName(QString nameParam);
    
    /**
     * get a widget of parameter board, getting with his name.
     * @param nameParam
     * @return 
     */
    QWidget *getParamWidgetByName(QString nameParam);

	PVFormatBuilderWidget* parent() { return _parent; }
    
    
private:
    QVBoxLayout *layout;
    Type type;
    QList<QVariant*> listOfParam;
    QList<QWidget *> lesWidgetDuLayout;
    QPushButton *btnApply;
    PVFormatBuilderWidget* _parent;
    QString pluginListURL;
    //bool confirmApply;
    PVRush::PVXmlTreeNodeDom  *nodeConfirmApply;
    QModelIndex editingIndex;
    
    QStringList listType(const QStringList&)const;
    QStringList getListTypeMapping(const QString& mType);
    QStringList getListTypePlotting(const QString& mType);
    
    
    
public slots:
    /**
     * This slot is called when we select a tree item.<br>
     * He draw the parameter box.
     * @param index
     */
    void edit(QModelIndex const& index);
    
    /**
     * Update de selection number with the regexp.
     * @param reg
     */
    void regExCountSel(const QString &reg);
    
    
    /**
     * For the axes parametter, this slot update the mapping and plotting choice.
     * @param type
     */
    void updatePlotMapping(const QString& );
    
    /**
     * Alert if we detecte a regexp in the name. <br> By star detection.
     */
    void slotConfirmRegExpInName(const QString &s);
        
    void slotEmitNeedApply();
    
    
    
    /**
     * This is onely used for the regex splitter editor, want editing the RegExp
     * whiche can create or delete some field.
     */
    void slotForceApply();
    
    void slotSelectNext();

signals:
    void signalNeedApply();
    void signalNeedConfirmApply( QModelIndex&);
    void signalForceApply(QModelIndex&);
    void signalSelectNext();
    void signalQuittingAParamBoard();

};
}

#endif	/* PVXMLPARAMWIDGET_H */

