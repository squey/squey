//! \file PVXmlEditorWidget.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef FEN2_H
#define	FEN2_H
#include <iostream>

#include <QTreeView>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include<QPushButton>
#include<QWidget>
#include<QToolBar>
#include <QFileDialog>
#include <QAction>
#include <QString>
#include <QMenuBar>
#include <QAbstractItemModel>
#include <QMainWindow>

#include <PVXmlDomModel.h>
#include <PVXmlTreeView.h>
#include <PVXmlParamWidget.h>
#include <pvrush/PVNormalizer.h>

namespace PVInspector{
class PVXmlEditorWidget : public QWidget{
    Q_OBJECT
public:
    PVXmlEditorWidget(QWidget * parent = NULL);

    virtual ~PVXmlEditorWidget();
private:
    PVXmlTreeView *myTreeView;
    PVXmlDomModel *myTreeModel;
    PVXmlParamWidget *myParamBord;
    
    QVBoxLayout *vbParam;
    QMenuBar *menuBar;
    
    void actionAllocation();
    
    /**
     * initialise les connexions dont tout les emitter/reciever sont des attributs
     * de la classe
     */
    void initConnexions();
    
    /**
     * init the menubar
     */
    void initMenuBar();
    
    
    /**
     * init the toolsbar
     * @param vb
     */
    void initToolBar(QVBoxLayout *vb);
    QAction *actionAddAxisAfter;
    QAction *actionAddAxisIn;
    QAction *actionAddFilterAfter;
    QAction *actionAddRegExAfter;
    QAction *actionAddRegExBefore;
    QAction *actionAddUrl;
    QAction *actionAddRegExIn;
    QPushButton *actionApply;
    QAction *actionDelete;
    QAction *actionMoveUp;
    QAction *actionMoveDown;
    QAction *actionOpenLog;
    QAction *actionOpen;
    QAction *actionSave;
    
  
    
    

public slots:

    //slots agissant sur l'arbre.
    void slotAddAxisIn();
    void slotAddFilterAfter();
    void slotAddRegExAfter();
    void slotAddUrl();
    void slotApplyModification();
    void slotDelete();
    void slotMoveUp();
    void slotMoveDown();
    void slotNeedApply();
    void slotOpen();
    void slotSave();
    void slotUpdateToolDesabled(const QModelIndex &);
};
}
#endif	/* FEN2_H */

