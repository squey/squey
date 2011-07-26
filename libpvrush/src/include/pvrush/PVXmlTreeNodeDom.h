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
#include <pvcore/PVArgument.h>
#include <pvfilter/PVFieldsFilterParamWidget.h>

#define trace_2(texte,texte2) { std::cout<<texte<<" "<<texte2<<std::endl; }


#define PVXmlTreeNodeDom_initXml "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n<!DOCTYPE PVParamXml>\n<param></param>\n"
    
namespace PVRush {
class PVXmlTreeNodeDom:public QObject {
    Q_OBJECT
public:
    enum Type {
        Root, field, RegEx, filter, axis, url, splitter
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

	void setFromArgumentList(PVCore::PVArgumentList const& args);
	
	void toArgumentList(PVCore::PVArgumentList const& default_args, PVCore::PVArgumentList& args);

    bool isEditable() {
        if (type == splitter || type == filter || type == url || type == axis || type == RegEx) {
            return true;
        } else {
            return false;
        }
    }
    


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

    void setSplitterPlugin(PVFilter::PVFieldsSplitterParamWidget_p plugin) {
        splitterPlugin = plugin;
		splitterPlugin->connect_to_args_changed(this, SLOT(slot_update()));
		splitterPlugin->connect_to_nchilds_changed(this, SLOT(slot_update_number_childs()));
    }

    PVFilter::PVFieldsSplitterParamWidget_p getSplitterPlugin() {
        if(!splitterPlugin){
            createSplitterPlugin(xmlDomElement);
			splitterPlugin->set_child_count(countChildren());
        }
        return splitterPlugin;
    }
    
    void createSplitterPlugin(const QDomElement &);
    
    QDomElement getDom();
    
	void updateFiltersDataDisplay();

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
    QString attribute(QString name, bool flagReadInXml=true);
    
    QWidget* getParamWidget(){
        PVCore::PVArgumentList args,args_default;
        args_default = getSplitterPlugin()->get_default_argument();
        toArgumentList(args_default,args);
        getSplitterPlugin()->get_filter()->set_args(args);
        return getSplitterPlugin()->get_param_widget();
    }
    
	void getChildrenFromField(PVCore::PVField const& field);
	void clearFiltersData();
    
    
    /**
     * définie un parent au node.
     * @param parent
     */
    void setParent(PVXmlTreeNodeDom *parent);

    
    void addRegExRacine();
    void addFilterRacine();
    
    void deleteFromTree();
    

    /**
     * add one field.
	 * @return the axis node that corresponds to that field
     */
	PVRush::PVXmlTreeNodeDom* addOneField(QString const& name);

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
    
    void version0to1();
    
    bool isOnRoot;

	QStringList getDataForRegexp() { return _data_for_regexp; }

	PVCol updateFieldLinearId(PVCol id);

	PVCol getFieldLinearId() const { return _field_linear_id; }

	PVXmlTreeNodeDom* getFirstFieldParent();
	
	bool hasSplitterAsChild();
    
private:
    QDomDocument xmlFile;
    QList<PVXmlTreeNodeDom*> children;
    PVXmlTreeNodeDom *parent;
    
    QDomElement xmlDomElement;
    QString str;
    
    QHash<QString,QString> otherData;
    
    PVFilter::PVFieldsSplitterParamWidget_p splitterPlugin;
    
    
    bool isAlreadyExplored;

    /**
     * method to explore child
     */
    void explore();
    
    /**
     * setup the type
     * @param nom
     */
    void setTypeFromString(const QString &nom);

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

	// AG: still the same saturday morning hack
	QStringList _data_for_regexp;

	// Id of a field, when the pipeline of filter is linearised. If this id equals to -1
	// it means that it has children !
	// TODO: list the ids of the children, so that they will be selected !
	ssize_t _field_linear_id;
    
public slots:
    void slot_update()
	{
        PVLOG_DEBUG("PVXmlTreeNodeDom slot slot_update()\n");
        setFromArgumentList(getSplitterPlugin()->get_filter()->get_args());
        emit data_changed();
    }

	void slot_update_number_childs()
	{
		assert(splitterPlugin);
		setNbr(splitterPlugin->get_child_count());
		emit data_changed();
	}

    signals:
    void data_changed();

    
};
}
#endif	/* NODEDOM_H */

