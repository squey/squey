/**
 * \file PVXmlDomModel.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef MONMODELE_H
#define	MONMODELE_H
#include <QAbstractItemModel>
#include <QDomDocument>
#include <QDomElement>

#include <QFile>
#include <QMessageBox>
#include <QTextStream>
#include <iostream>
#include <QDebug>
#include <QString>
#include <QSet>

#include <pvkernel/rush/PVXmlTreeNodeDom.h>
#include <PVXmlParamWidget.h>
#include <pvkernel/rush/PVXmlParamParser.h>
#include <pvkernel/filter/PVFieldsFilterParamWidget.h>
#include <picviz/PVAxesCombination.h>

namespace PVInspector{
class PVXmlDomModel: public QAbstractItemModel {
    
    Q_OBJECT

public:
    PVXmlDomModel(QWidget * parent = NULL);
    virtual ~PVXmlDomModel();
            
    /*
     * Toolbar methods
     */
    void addFilterAfter(QModelIndex &index);
    void applyModification(QModelIndex &index,PVXmlParamWidget *paramBord);
    
    /*
    * Add items
    */
    void addAxisIn(const QModelIndex &index);
    PVRush::PVXmlTreeNodeDom* addAxisIn(PVRush::PVXmlTreeNodeDom* parentNode);

	PVRush::PVXmlTreeNodeDom* addSplitter(const QModelIndex &index, PVFilter::PVFieldsSplitterParamWidget_p splitterPlugin);
	PVRush::PVXmlTreeNodeDom* addConverter(const QModelIndex &index, PVFilter::PVFieldsConverterParamWidget_p converterPlugin);
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
    
    
    bool saveXml(QString xml_file);
    
    void deleteSelection(QModelIndex const& index);
    
    void moveDown(const QModelIndex &index);
    void moveUp(const QModelIndex &index);
    QModelIndex selectNext(const QModelIndex &index);
    
    
    //open a pcre
    bool openXml(QString file);
	void openXml(QDomDocument& doc);
    
    //identify multi axis or splitter in a field
    bool trustConfictSplitAxes(const QModelIndex &index);
    
    //find level count form index to parent
    int countParent(const QModelIndex &index);
    
    PVRush::PVXmlTreeNodeDom *nodeFromIndex(const QModelIndex &index)const;

	QDomElement const& getRootDom() const { return xmlRootDom; }

	PVRush::PVXmlTreeNodeDom* getRoot() { return rootNode; }

	void processChildrenWithField(PVCore::PVField const& field);
	void clearFiltersData();
	void updateFieldsLinearId();
	void updateFiltersDataDisplay();
	void setAxesNames(QStringList const& names);
	void updateAxesCombination();

	PVRush::types_groups_t& getGroups() { return _groups; }

	Picviz::PVAxesCombination& get_axes_combination() { return _axes_combination; }
    
private:
	static void setDefaultAttributesForAxis(QDomElement& elt);
	void setEltMappingPlotting(QDomElement& elt, QString const& type, QString const& mode_mapping, QString const& mode_plotting);

private:
    PVRush::PVXmlTreeNodeDom *rootNode;
    
    QString urlXml;
    QDomDocument xmlFile;
    QDomElement xmlRootDom;

	// types_groups_t defined in pvkernel/rush/PVXmlTreeNodeDom.h
	PVRush::types_groups_t _groups;
    
	Picviz::PVAxesCombination _axes_combination;
    
public slots:
    void saveDefault();


};
}
#endif	/* MONMODELE_H */

