//! \file PVXmlParamWidgetBoardAxis.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVXMLPARAMWIDGETBOARDAXIS_H
#define	PVXMLPARAMWIDGETBOARDAXIS_H
#include <QWidget>
#include <QDir>
#include <QStringList>
#include <QRegExp>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QVariant>
#include <QDebug>
#include <QTableWidget>
#include <QTextEdit>
#include <QDateTime>
#include <QPushButton>
#include <QGroupBox>
#include <QWebPage>
#include <QWebFrame>
#include <QTabWidget>
#include <QCheckBox>

#include <pvkernel/rush/PVXmlTreeNodeDom.h>
#include <PVXmlParamWidgetEditorBox.h>
#include <PVXmlParamTextEdit.h>
#include <PVXmlParamComboBox.h>
#include <PVXmlParamColorDialog.h>
#include <picviz/plugins.h>
namespace PVInspector{
class PVXmlParamWidgetBoardAxis : public QWidget {
    Q_OBJECT
public:
    PVXmlParamWidgetBoardAxis(PVRush::PVXmlTreeNodeDom *pNode);
    virtual ~PVXmlParamWidgetBoardAxis();
    QWidget *getWidgetToFocus();
    
  private:
    void allocBoardFields();
    QVBoxLayout *createTab(const QString &title, QTabWidget *tab);
    void disableConnexion();
    void disAllocBoardFields();
    void draw();
    void initConnexion();
    void initValue();
    void setHelp();
    
    
    
    QStringList listType(const QStringList &listEntry)const;
    QStringList getListTypeMapping(const QString& mType);
    QStringList getListTypePlotting(const QString& mType);
    
    
    
    /***************************  board items **********************/
    //***** tab general ***** 
    QTabWidget *tabParam;
    PVXmlParamWidgetEditorBox *textName;//name
    //type
    PVXmlParamComboBox * mapPlotType;
    PVXmlParamComboBox * comboMapping;
    PVXmlParamComboBox * comboPlotting;
    
    //***** tab time format ***** 
    QLabel *timeFormatLabel;
    PVXmlParamTextEdit *timeFormat;
    PVXmlParamTextEdit *timeFormatInTab;
    QString timeFormatStr;
    QGroupBox * groupTimeFormatValidator;
    QTextEdit *helpTimeFormat;
    PVXmlParamTextEdit *timeSample;
    QCheckBox *useParentRegExpValue;
    
    //***** tab param ***** 
    PVXmlParamComboBox * comboKey;
    QLabel *keyLabel;
    PVXmlParamWidgetEditorBox *group;
    QLabel *groupLabel;
    PVXmlParamColorDialog *buttonColor;
    QLabel *colorLabel;
    PVXmlParamColorDialog *buttonTitleColor;
    QLabel *titleColorLabel;
    
    //***** view values from parent regexp *****
    QTextEdit *tableValueFromParentRegExp;
    
    QPushButton *buttonNextAxis;
    /***************************  board items **********************/
    
    
    //editing node
    PVRush::PVXmlTreeNodeDom *node;
    QString pluginListURL;
    
public slots:
    void slotGoNextAxis();
    void slotSetValues();
    void updatePlotMapping(const QString& t) ;
    void updateDateValidation();
    //void slotSetVisibleExtra(bool flag);
    void slotSetVisibleTimeValid(bool flag);
    
    signals:
    void signalRefreshView();
    void signalSelectNext();
};
}
#endif	/* PVXMLPARAMWIDGETBOARDAXIS_H */

