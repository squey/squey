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

#include <PVXmlTreeNodeDom.h>
#include <PVXmlParamWidget.h>
#include <pvcore/PVXmlParamParser.h>

#define message(string){QMessageBox qb;   qb.setText(string);    qb.exec();} 
//#define dbg {qDebug()<<__FILE__<<__LINE__;}
namespace PVInspector{
class PVXmlDomModel: public QAbstractItemModel {
    
    Q_OBJECT
public:
    PVXmlDomModel();
    
    /**
     * @param url path of the XML file
     */
    PVXmlDomModel(QString url);
    virtual ~PVXmlDomModel();

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
     * initialisation du node root.
     * @param 
     */
    void setRoot(PVXmlTreeNodeDom *);
    
    /**
     * Récupère l'objet Dom en fonction de l'indexe.
     * @param QModelIndex index
     * @return QDomElement* objetDom
     */
    QDomElement *getItem(QModelIndex &index);
    
    
    void saveXml(QString fichierXml);
    
    
    /*
     * Méthodes for the toolBar
     */
    void addFilterAfter(QModelIndex &index);
    void addRegExAfter(QModelIndex &index);
    void applyModification(QModelIndex &index,PVXmlParamWidget *paramBord);
    
    /*
    * Add items
    */
    void addAxisIn(const QModelIndex &index);
    void addFisrtRegEx();
    void addFirstFilter();
    void addRegExIn(const QModelIndex &index);
    void addUrlIn(const QModelIndex &index);
    
    
    void deleteSelection(QModelIndex &index);
    
    void moveDown(const QModelIndex &index);
    void moveUp(const QModelIndex &index);
    QModelIndex selectNext(const QModelIndex &index);
    
    
    //open a pcre
    void openXml(QString);
    
    //identify multi axis or splitter in a field
    bool trustConfictSplitAxes(const QModelIndex &index);
    
    //find level count form index to parent
    int countParent(const QModelIndex &index);
    
    PVXmlTreeNodeDom *nodeFromIndex(const QModelIndex &index)const;
    
private:
    PVXmlTreeNodeDom *rootNode;
    
    QString urlXml;
    QDomDocument xmlFile;
    QDomElement xmlRootDom;
    
    
public slots:
    void saveDefault();


};
}
#endif	/* MONMODELE_H */

