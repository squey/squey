//! \file PVXmlParamWidgetBoardSplitterRegEx.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVXMLPARAMWIDGETBOARDSPLITTERREGEX_H
#define	PVXMLPARAMWIDGETBOARDSPLITTERREGEX_H
#include <QWidget>
#include <QDir>
#include <QFileDialog>
#include <QStringList>
#include <QRegExp>
#include <QVBoxLayout>
#include <QLabel>
#include <QVariant>
#include <QDebug>
#include <QGroupBox>
#include <QFrame>
#include <QStyle>
#include <QCheckBox>
#include <QTableWidget>
#include <QFile>
#include <QTableWidgetItem>
#include <QAbstractItemModel>


#include <PVXmlTreeNodeDom.h>
#include <PVXmlParamWidgetEditorBox.h>
#include <PVXmlParamTextEdit.h>
#include <PVXmlParamComboBox.h>
#include <PVXmlParamColorDialog.h>
namespace PVInspector{
class PVXmlParamWidgetBoardSplitterRegEx : public QWidget {
    Q_OBJECT
public:
    PVXmlParamWidgetBoardSplitterRegEx(PVXmlTreeNodeDom *pNode);
    virtual ~PVXmlParamWidgetBoardSplitterRegEx();
    
    bool confirmAndSave();
    QWidget *getWidgetToFocus();
    

private:
    void allocBoardFields();
    QVBoxLayout *createTab(const QString &title, QTabWidget *tab);
    void disableConnexion();
    void disAllocBoardFields();
    void draw();
    void initConnexion();
    void initValue();
    void updateHeaderTable();
    
    //field
    QTabWidget *tabParam;
    PVXmlParamWidgetEditorBox *name;
    PVXmlParamWidgetEditorBox *exp;
    PVXmlParamTextEdit *validWidget;
    QCheckBox *checkSaveValidLog;
    QCheckBox *checkUseTable;
    QTableWidget *table;
    QPushButton *openLog;
    QLabel *labelNbr;
    int nbr;
    QPushButton *btnApply;
    bool useTableVerifie();

    //editing node
    PVXmlTreeNodeDom *node;
    bool flagNeedConfirmAndSave;
    bool flagAskConfirmActivated;
    bool flagSaveRegExpValidator;
    
public slots:
    void slotNoteConfirmationNeeded();
    void slotOpenLogValid();
    void slotSaveValidator(bool);
    void slotSetValues();
    void slotSetConfirmedValues();
    void slotShowTable(bool);
    /**
    * Each time we modify the name, we detect if it's a regexp.<br>
    * Then we propose to push it in the regexp field.
    */
    void slotVerifRegExpInName();
    /**
    * This slot update the table where we can see the selection
    */
    void slotUpdateTable();
    void regExCount(const QString &reg);
    void exit();


signals:
    void signalRefreshView();
    void signalNeedConfirmAndSave();
};
}
#endif	/* PVXMLPARAMWIDGETBOARDSPLITTERREGEX_H */

