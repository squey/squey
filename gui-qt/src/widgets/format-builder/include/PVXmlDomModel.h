//! \file PVXmlDomModel.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef MONMODELE_H
#define	MONMODELE_H
#include <QAbstractItemModel>
#include <QDomDocument>
#include <QDomElement>

#include <QFile>
#include <QMessageBox>
#include <QTextStream>
#include <Qt>
#include <iostream>
#include <QDebug>
#include <QString>

#include <pvrush/PVXmlTreeNodeDom.h>
#include <PVXmlParamWidget.h>
#include <pvrush/PVXmlParamParser.h>
#include <pvfilter/PVFieldsFilterParamWidget.h>

#define FORMAT_VERSION 1.0

#define message(string){QMessageBox qb;   qb.setText(string);    qb.exec();} 
//#define dbg {qDebug()<<__FILE__<<__LINE__;}
namespace PVInspector{
class PVXmlDomModel: public QAbstractItemModel {
    
    Q_OBJECT
public:
    PVXmlDomModel(QWidget * parent = NULL);
    
    /**
     * @param url path of the XML file
     */
    PVXmlDomModel(QString url);
    virtual ~PVXmlDomModel();
    
    
    
    /*
     * Méthodes for the toolBar
     */
    void addFilterAfter(QModelIndex &index);
    void applyModification(QModelIndex &index,PVXmlParamWidget *paramBord);
    
    /*
    * Add items
    */
    void addAxisIn(const QModelIndex &index);
	PVRush::PVXmlTreeNodeDom* addAxisIn(PVRush::PVXmlTreeNodeDom* parentNode);

	PVRush::PVXmlTreeNodeDom* addSplitter(const QModelIndex &index, PVFilter::PVFieldsSplitterParamWidget_p splitterPlugin);
    void addRegExIn(const QModelIndex &index);
    void addUrlIn(const QModelIndex &index);

	PVRush::PVXmlTreeNodeDom* addSplitterWithAxes(const QModelIndex& index, PVFilter::PVFieldsSplitterParamWidget_p splitterPlugin, QStringList axesName);

    
    /*
     * virtual method from QAbstractItemModel
     */
    virtual QModelIndex index(int, int, const QModelIndex&)const;
    virtual QModelIndex parent(const QModelIndex&)const;
    virtual int rowCount(const QModelIndex&)const;
    virtual int columnCount(const QModelIndex&)const;
    virtual QVariant data(const QModelIndex&, int)const;

    //return selectable
    Qt::ItemFlags flags(const QModelIndex & index) const;
    
    //drag&drop
    Qt::DropActions supportedDropActions() const;

    //respond to the View
    bool setData(const QModelIndex & index, const QVariant & value, int role = Qt::EditRole);

    
    
    
    /**
     * initialisation of the root.
     * @param 
     */
    void setRoot(PVRush::PVXmlTreeNodeDom *);
    
    /**
     * get the Dom with index.
     * @param QModelIndex index
     * @return QDomElement* objetDom
     */
    QDomElement *getItem(QModelIndex &index);
    
    /**
     * get the version of the format
     * @return 
     */
    QString getVersion(){return xmlRootDom.attribute("version","0");}
    void setVersion(QString v){ xmlRootDom.setAttribute("version",v);}
    
    /**
     * 
     * @param section : raw or col index
     * @param orientation : header vertical or not
     * @param role : what we are doing
     * @return something to write on tree header
     */
    virtual QVariant headerData ( int section, Qt::Orientation orientation, int role = Qt::DisplayRole ) const;
    
    
    bool saveXml(QString fichierXml);
    
    void deleteSelection(QModelIndex &index);
    
    void moveDown(const QModelIndex &index);
    void moveUp(const QModelIndex &index);
    QModelIndex selectNext(const QModelIndex &index);
    
    
    //open a pcre
    bool openXml(QString);
    
    //identify multi axis or splitter in a field
    bool trustConfictSplitAxes(const QModelIndex &index);
    
    //find level count form index to parent
    int countParent(const QModelIndex &index);
    
    PVRush::PVXmlTreeNodeDom *nodeFromIndex(const QModelIndex &index)const;

	QDomElement const& getRootDom() const { return xmlRootDom; }
    
private:
	static void setDefaultAttributesForAxis(QDomElement& elt);

private:
    PVRush::PVXmlTreeNodeDom *rootNode;
    
    QString urlXml;
    QDomDocument xmlFile;
    QDomElement xmlRootDom;
    
    
public slots:
    void saveDefault();


};
}
#endif	/* MONMODELE_H */

