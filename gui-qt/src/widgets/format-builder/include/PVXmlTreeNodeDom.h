//! \file PVXmlTreeNodeDom.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef NODEDOM_H
#define	NODEDOM_H

#include <QObject>
#include<QString>
#include<QStringList>
#include<QList>
#include<QDomElement>
#include<QDomDocument>
#include<QDebug>

#include<iostream>

#include <pvcore/general.h>
#define trace_2(texte,texte2) { std::cout<<texte<<" "<<texte2<<std::endl; }


#define PVXmlTreeNodeDom_initXml "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n<!DOCTYPE PVParamXml>\n<param></param>\n"
    
namespace PVInspector{
class PVXmlTreeNodeDom:public QObject {
    Q_OBJECT;
public:
    enum Type {
        Root, field, RegEx, filter, axis, url
    };
    
    
    PVXmlTreeNodeDom();
    PVXmlTreeNodeDom(QDomElement const& dom);

    
    /**
     * Constructor defining the name for le node.
     * @param _type
     * @param _str
     */
    PVXmlTreeNodeDom(Type _type, const QString &_str,  QDomElement &dom, QDomDocument &file);
    virtual ~PVXmlTreeNodeDom();
    
    /**
     * Add a child.
     * @param child
     */
    void addChild(PVXmlTreeNodeDom *child);
    /**
     * Add a child specifying the row index.
     * @param child
     * @param row
     */
    void addChildAt(PVXmlTreeNodeDom *child, int row);
    
    /**
     * Remove a child
     * @param child
     */
    void removeChild(PVXmlTreeNodeDom *child);
    
    /**
     * Return the child from index row.
     * @param i
     * @return the child
     */
    PVXmlTreeNodeDom* getChild(int i);
    
    /**
     * Return the list with the children.
     * @return children list
     */
    QList<PVXmlTreeNodeDom*> getChildren();
    
    /**
     * Return the children count.
     * @return number
     */
    int countChildren();
    
    /**
     * Return the parent node.
     * @return parent
     */
    PVXmlTreeNodeDom *getParent();
    
    
    /**
     * Return index row in the child list of parent.
     * @return 
     */
    int getRow();
    
    
    /**
     * Add a new filter, after the selected item.
     * @param indexeThisParmisFrere
     */
    void newFilterAfter(int indexeThisParmisFrere);
    
    /**
     * Add a new RegEx after the selected item.
     * @param indexeThisParmisFrere
     */
    void newSplitterAfter(int indexeThisParmisFrere, QString type="RegEx");

    /**
     * Setup the DomDocument reference.
     * @param DomDocument
     */
    void setDoc(QDomDocument &file);
    
    
    Type type;
    
    

    QString getName();
    void setName(QString nom);
    

    QString getExpression();
    void setExpression(QString exp);
    

    int getNbr();
    void setNbr(int nbr);
    
    QDomElement getDom();
    
    /**
     * General attribute setter.
     * @param name
     * @param Value
     */
    void setAttribute(QString name, QString Value, bool flagSaveInXml=true);
    /**
     * General attribute getter.
     * @param name
     * @return 
     */
    QString getAttribute(QString name, bool flagReadInXml=true);
    
    
    
    /**
     * définie un parent au node.
     * @param parent
     */
    void setParent(PVXmlTreeNodeDom *parent);

    
    void addRegExRacine();
    void addFilterRacine();
    
    void deleteFromTree();
    
    /**
     * Return the type of node in a QString.
     * @return type
     */
    QString typeToString();
    
    /**
     * return the name of axis regexp or url name of each field.
     * @return 
     */
    QString getOutName();
    /**
     * return the node of axis regexp or url name of each field.
     * @return 
     */
    PVXmlTreeNodeDom *getOutWidget();
    
    bool isOnRoot;
    
private:
    QDomDocument xmlFile;
    QList<PVXmlTreeNodeDom*> children;
    PVXmlTreeNodeDom *parent;
    
    QDomElement xmlDomElement;
    QString str;
    
    QHash<QString,QString> otherData;
    
    
    
    
    bool isAlreadyExplored;

    /**
     * method to explore child
     */
    void explore();
    
    /**
     * setup the type
     * @param nom
     */
    void setTypeFromString(QString nom);

    /**
     * add 'n' field.
     * @param n
     */
    void addField(int n);
    
    /**
     * delete 'n' field.
     * @param n
     */
    void delField(int n);
    
    bool isFieldOfUrl();
    
};
}
#endif	/* NODEDOM_H */

