/**
 * \file PVXmlParamWidgetBoardSplitterRegEx.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

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


#include <pvkernel/rush/PVXmlTreeNodeDom.h>
#include <PVXmlParamWidgetEditorBox.h>
#include <PVXmlParamTextEdit.h>
#include <PVXmlParamComboBox.h>
#include <PVXmlParamColorDialog.h>
namespace PVInspector{

class PVXmlParamWidget;

class PVXmlParamWidgetBoardSplitterRegEx : public QWidget {
    Q_OBJECT
public:
    PVXmlParamWidgetBoardSplitterRegEx(PVRush::PVXmlTreeNodeDom *pNode, PVXmlParamWidget* parent);
    virtual ~PVXmlParamWidgetBoardSplitterRegEx();
    
    bool confirmAndSave();
    QWidget *getWidgetToFocus();
    
	void setData(QStringList const& data)
	{
		for (int i = 0; i < data.size(); i++) {
			PVLOG_INFO("(regexp board widget) get data %s\n", qPrintable(data[i]));
		}
		_data = data;

		QString textVal;
		for (int i = 0; i < _data.size(); i++) {
			textVal += _data[i];
			textVal += QChar('\n');
		}

		validWidget->setVal(textVal);
	}
    

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
	PVXmlParamWidget* _parent;

    //editing node
    PVRush::PVXmlTreeNodeDom *node;
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

protected:
	QStringList _data;
};
}
#endif	/* PVXMLPARAMWIDGETBOARDSPLITTERREGEX_H */

