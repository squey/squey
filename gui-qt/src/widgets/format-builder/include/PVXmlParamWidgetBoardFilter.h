/**
 * \file PVXmlParamWidgetBoardFilter.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVXMLPARAMWIDGETBOARDFILTER_H
#define	PVXMLPARAMWIDGETBOARDFILTER_H
#include <QWidget>
#include <QDir>
#include <QStringList>
#include <QRegExp>
#include <QVBoxLayout>
#include <QLabel>
#include <QVariant>
#include <QDebug>

#include <pvkernel/rush/PVXmlTreeNodeDom.h>
#include <PVXmlParamWidgetEditorBox.h>
#include <PVXmlParamTextEdit.h>
#include <PVXmlParamComboBox.h>
#include <PVXmlParamColorDialog.h>

namespace PVInspector{

class PVXmlParamWidget;

class PVXmlParamWidgetBoardFilter : public QWidget {
    Q_OBJECT
public:
    PVXmlParamWidgetBoardFilter(PVRush::PVXmlTreeNodeDom *node, PVXmlParamWidget* parent);
    virtual ~PVXmlParamWidgetBoardFilter();
    QWidget *getWidgetToFocus();
    
    
private:
    void allocBoardFields();
    void disableConnexion();
    void disAllocBoardFields();
    void draw();
    void initConnexion();
    void initValue();
    
    
    //field
    PVXmlParamWidgetEditorBox *name;
    PVXmlParamWidgetEditorBox *exp;
    PVXmlParamTextEdit *validWidget;
    PVXmlParamComboBox *typeOfFilter;
    QPushButton *buttonNext;


    //editing node
    PVRush::PVXmlTreeNodeDom *node;

	PVXmlParamWidget* _parent;

public slots:
    void slotSetValues();
    void slotVerifRegExpInName();
    void slotEmitNext();

signals:
    void signalRefreshView();
    void signalEmitNext();
};
}
#endif	/* PVXMLPARAMWIDGETBOARDFILTER_H */

