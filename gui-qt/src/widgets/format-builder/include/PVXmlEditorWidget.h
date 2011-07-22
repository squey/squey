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
#include <QDomElement>
#include <QDomDocument>

#include <PVXmlDomModel.h>
#include <PVXmlTreeView.h>
#include <PVXmlParamWidget.h>
#include <PVNrawListingWidget.h>
#include <PVNrawListingModel.h>
#include <pvrush/PVNormalizer.h>
#include <pvcore/PVRegistrableClass.h>
#include <pvcore/PVClassLibrary.h>
#include <pvcore/PVArgument.h>
#include <pvfilter/PVFieldsFilterParamWidget.h>
#include <pvfilter/PVRawSourceBase.h>
#include <pvrush/PVSourceCreator.h>
#include <pvrush/PVExtractor.h>
#include <pvrush/PVInputType.h>

namespace PVInspector{

typedef QList<PVFilter::PVFieldsSplitterParamWidget_p> list_splitters_t;
typedef QList<PVFilter::PVFieldsFilterParamWidget<PVFilter::one_to_one> > list_filters_t;

class PVXmlEditorWidget : public QWidget{
    Q_OBJECT
public:
    PVXmlEditorWidget(QWidget * parent = NULL);

    virtual ~PVXmlEditorWidget();
private:
    //
    PVXmlTreeView *myTreeView;
    PVXmlDomModel *myTreeModel;
    PVXmlParamWidget *myParamBord_old_model;
    QWidget *myParamBord;
    QWidget emptyParamBoard;
    //
    QVBoxLayout *vbParam;
    QMenuBar *menuBar;
    //
    QFile logFile;///!< file we open to edit the format
    int lastSplitterPluginAdding;
    
    
    void actionAllocation();
    
    void hideParamBoard();
    
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
    QAction *actionOpen;
    QAction *actionSave;
    
    /**
     * init the splitters list, by listing the plugins found
     */
    void initSplitters();    
	list_splitters_t _list_splitters;///!<list of the plugins splitters
	list_filters_t _list_filters;///!<list of the plugins filters
    
    void showParamBoard(PVRush::PVXmlTreeNodeDom *node);
    

// Log input management

protected:
	void update_table(PVRow start, PVRow end);
	void set_format_from_dom();
	void create_extractor();
	void guess_first_splitter();
	bool is_dom_empty();

protected:
	PVCore::PVArgument _log_input;
	PVRush::PVInputType_p _log_input_type;
	PVRush::PVSourceCreator_p _log_sc;
	PVFilter::PVRawSourceBase_p _log_source;
	boost::shared_ptr<PVRush::PVExtractor> _log_extract; 

	// Model and widget for the NRAW
	PVNrawListingModel* _nraw_model;
	PVNrawListingWidget* _nraw_widget;
    

public slots:

    //slots agissant sur l'arbre.
    void slotAddAxisIn();
    void slotAddFilterAfter();
    void slotAddRegExAfter();
    void slotAddSplitter();
    void slotAddUrl();
    void slotApplyModification();
    void slotDelete();
    void slotMoveUp();
    void slotMoveDown();
    void slotNeedApply();
    void slotOpen();
    void slotOpenLog();
    void slotSave();
    void slotUpdateToolDesabled(const QModelIndex &);
	void slotExtractorPreview();
};

}
#endif	/* FEN2_H */

