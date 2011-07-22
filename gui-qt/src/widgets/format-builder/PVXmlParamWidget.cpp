///! \file PVXmlParamWidget.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011







#include <PVXmlParamWidget.h>
#include <PVXmlParamWidgetEditorBox.h>
#include <PVXmlParamComboBox.h>

#include <PVXmlParamTextEdit.h>

#define dbg {qDebug()<<__FILE__<<":"<<__LINE__;}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::PVXmlParamWidget
 *
 *****************************************************************************/
PVInspector::PVXmlParamWidget::PVXmlParamWidget() : QWidget() {
    layout = new QVBoxLayout();
    setObjectName("PVXmlParamWidget");

    //confirmApply = false;
    //listeDeParam = new QList<QVariant*>();
    removeListWidget();
    type = no;
    layout->setContentsMargins(0,0,0,0);
    setLayout(layout);
    pluginListURL = picviz_plugins_get_functions_dir();
}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::~PVXmlParamWidget
 *
 *****************************************************************************/
PVInspector::PVXmlParamWidget::~PVXmlParamWidget() {
    layout->deleteLater();
}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::drawForNo
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::drawForNo(QModelIndex) {

    //confirmApply = false;
    emit signalQuittingAParamBoard();
    removeListWidget();
    type = no;
}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::drawForAxis
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::drawForAxis(PVRush::PVXmlTreeNodeDom *nodeOnClick) {
    PVXmlParamWidgetBoardAxis *axisboard = new PVXmlParamWidgetBoardAxis(nodeOnClick);
    lesWidgetDuLayout.push_back(axisboard);
    layout->addWidget(axisboard);
    connect(axisboard, SIGNAL(signalRefreshView()), this, SLOT(slotEmitNeedApply()));
    connect(axisboard, SIGNAL(signalSelectNext()), this, SLOT(slotSelectNext()));
    type = filterParam;
    //focus on name
    axisboard->getWidgetToFocus()->setFocus();
}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::drawForFilter
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::drawForFilter(PVRush::PVXmlTreeNodeDom *nodeFilter) {
  
    PVXmlParamWidgetBoardFilter *filterboard = new PVXmlParamWidgetBoardFilter(nodeFilter);
    lesWidgetDuLayout.push_back(filterboard);
    layout->addWidget(filterboard);
    connect(filterboard, SIGNAL(signalRefreshView()), this, SLOT(slotEmitNeedApply()));
    connect(filterboard, SIGNAL(signalEmitNext()), this, SLOT(slotSelectNext()));
    type = filterParam;
    
    //focus name.
    filterboard->getWidgetToFocus()->setFocus();
}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::drawForRegEx
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::drawForRegEx(PVRush::PVXmlTreeNodeDom *nodeSplitter) {
    PVXmlParamWidgetBoardSplitterRegEx *regExpBoard = new PVXmlParamWidgetBoardSplitterRegEx(nodeSplitter);
    lesWidgetDuLayout.push_back(regExpBoard);
    layout->addWidget(regExpBoard);
    connect(regExpBoard, SIGNAL(signalRefreshView()), this, SLOT(slotEmitNeedApply()));
    connect(this, SIGNAL(signalQuittingAParamBoard()),regExpBoard,SLOT(exit()));
    
    addListWidget();
    type = filterParam;
    //focus on regexp
    regExpBoard->getWidgetToFocus()->setFocus();
}
/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::drawForSplitter
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::drawForSplitter(PVRush::PVXmlTreeNodeDom *nodeSplitter) {
        assert(nodeSplitter);
        assert(nodeSplitter->getSplitterPlugin());
        QWidget *w = nodeSplitter->getParamWidget();
        lesWidgetDuLayout.push_back(w);
        layout->addWidget(w);
        addListWidget();
        type = splitterParam;
        
        connect(nodeSplitter, SIGNAL(data_changed()), this, SLOT(slotEmitNeedApply()));
        //slotEmitNeedApply();
        //focus on regexp
        //w->getWidgetToFocus()->setFocus();
}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::addListWidget
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::addListWidget() {
    for (int i = 0; i < lesWidgetDuLayout.count(); i++) {
        if (i == 2) {
            layout->addSpacerItem(new QSpacerItem(1, 1, QSizePolicy::Minimum, QSizePolicy::Expanding));
        }
        if (lesWidgetDuLayout.at(i)->objectName() != "nbr")
            layout->addWidget(lesWidgetDuLayout.at(i));
    }
    layout->addSpacerItem(new QSpacerItem(1, 1, QSizePolicy::Minimum, QSizePolicy::Expanding));
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::removeListWidget
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::removeListWidget() {
        /*
         * Suppression de tous les widgets (textes, btn, editBox...).
         */
        while (!lesWidgetDuLayout.isEmpty()) {
                QWidget *tmp = lesWidgetDuLayout.front();

                tmp->hide();
                tmp->close();
                layout->removeWidget(tmp);
                lesWidgetDuLayout.removeFirst();
                layout->update();
        }

        /* 
         *  Suppression des items (commes les spacers).
         */
        for (int i = 0; i < 10; i++) {
                layout->removeItem(layout->itemAt(0));
        }
}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::getParam
 *
 *****************************************************************************/
QVariant PVInspector::PVXmlParamWidget::getParam(int i) {
    int id;
    if (i == 0){
      id = 1;
    }else if (i == 1){
      id = 3;
    }else if (i == 2){
      id = 5;
    }else {
      return QVariant();
    }
    return ((PVXmlParamWidgetEditorBox*) lesWidgetDuLayout.at(id))->val();
}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::getParamVariantByName
 *
 *****************************************************************************/
QVariant PVInspector::PVXmlParamWidget::getParamVariantByName(QString nameParam) {
    for (int i = 0; i < lesWidgetDuLayout.count(); i++) {
        QWidget *w = ((QWidget*) lesWidgetDuLayout.at(i));
        if (w->objectName() == nameParam) {
            if (nameParam == "validator" || nameParam == "time-format") {
                return ((PVXmlParamTextEdit*) w)->getVal();
            } else if (nameParam == "typeCombo" || nameParam == "mapping" || nameParam == "plotting" || nameParam == "key") {
                return ((PVXmlParamComboBox*) w)->val();
            } else if (nameParam == "color" || nameParam == "titlecolor") {
                return ((PVXmlParamColorDialog*) w)->getColor();
            } else
                return ((PVXmlParamWidgetEditorBox*) w)->val();
        }
    }
    return QVariant();
}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::getParamWidgetByName
 *
 *****************************************************************************/
QWidget * PVInspector::PVXmlParamWidget::getParamWidgetByName(QString nameParam) {
    for (int i = 0; i < lesWidgetDuLayout.count(); i++) {
        QWidget *w = ((QWidget*) lesWidgetDuLayout.at(i));
        if (w->objectName() == nameParam) {
            return w;
        }
    }
    return NULL;
}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::listType
 *
 *****************************************************************************/
QStringList PVInspector::PVXmlParamWidget::listType(const QStringList &listEntry)const {
    //listEntry.filter("(.*function_mapping.*)");//no
    QStringList ll;
    QRegExp reg(".*function_mapping_([a-zA-Z0-9\\-]*)_.*");

    for (int i = 0; i < listEntry.count(); i++) {
        if (reg.exactMatch(listEntry.at(i))) {
            reg.indexIn(listEntry.at(i));
            if (!ll.contains(reg.cap(1)))
                ll.push_back(reg.cap(1));
        }
    }
    return ll;
}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::getListTypeMapping
 *
 *****************************************************************************/
QStringList PVInspector::PVXmlParamWidget::getListTypeMapping(const QString& mType) {
    QDir dir(pluginListURL);
    QStringList lm;
    QStringList fileList = dir.entryList();

    QString regexpStr = QString(".*function_mapping_%1_([a-zA-Z0-9\\-]*)\\..*").arg(mType);
    QRegExp reg(regexpStr);

    for (int i = 0; i < fileList.count(); i++) {
        if (reg.exactMatch(fileList.at(i))) {
            reg.indexIn(fileList.at(i));
            if (!lm.contains(reg.cap(1)))
                lm.push_back(reg.cap(1));
        }
    }
    return lm;
}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::getListTypePlotting
 *
 *****************************************************************************/
QStringList PVInspector::PVXmlParamWidget::getListTypePlotting(const QString& mType) {
    QDir dir(pluginListURL);
    //qDebug() << "updatePlotMapping(" << mType << ")";
    QStringList lp;
    QStringList fileList = dir.entryList();

    QString regexpStr = QString(".*function_plotting_%1_([a-zA-Z0-9\\-]*)\\..*").arg(mType);
    QRegExp reg(regexpStr);

    for (int i = 0; i < fileList.count(); i++) {
        if (reg.exactMatch(fileList.at(i))) {
            reg.indexIn(fileList.at(i));
            if (!lp.contains(reg.cap(1)))
                lp.push_back(reg.cap(1));
        }
    }
    return lp;
}





/******************************** SLOTS ***************************************/


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::edit
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::edit(QModelIndex const& index) {

    drawForNo(index);
    if (index.isValid()) {
        //emit signalQuittingAParamBoard();
        //if (confirmApply == true) {
//            //emit the signal to require confirmation.
        //    emit signalNeedConfirmApply(editingIndex);
        //}
        //confirmApply = false;
        editingIndex = index;
        PVRush::PVXmlTreeNodeDom *nodeOnClick = (PVRush::PVXmlTreeNodeDom *) index.internalPointer();
        
     
        
        if (nodeOnClick->type == PVRush::PVXmlTreeNodeDom::filter) {
	  drawForFilter(nodeOnClick);
	}else
        if (nodeOnClick->type == PVRush::PVXmlTreeNodeDom::RegEx) {
            drawForRegEx(nodeOnClick);
            //confirmApply = false;
        }else
        if (nodeOnClick->type == PVRush::PVXmlTreeNodeDom::axis){
	  drawForAxis(nodeOnClick);
	}else
        if (nodeOnClick->type == PVRush::PVXmlTreeNodeDom::splitter){
	  drawForSplitter(nodeOnClick);
	}
    }

}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::slotForceApply
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::slotForceApply() {
    emit signalForceApply(editingIndex);
}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::regExCountSel
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::regExCountSel(const QString &reg) {
    QRegExp regExp = QRegExp(reg);
    PVXmlParamWidgetEditorBox *widgetNumberOfSelect = (PVXmlParamWidgetEditorBox *) getParamWidgetByName("nbr");
    //update number of selection count
    widgetNumberOfSelect->setVal(QVariant(regExp.captureCount()));
}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::updatePlotMapping
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::updatePlotMapping(const QString& t) {
    //qDebug() << "updatePlotMapping(" << t << ")";
    if (t.length() > 1) {

        PVXmlParamComboBox *myType = (PVXmlParamComboBox *) getParamWidgetByName("typeCombo");
        PVXmlParamComboBox *myPlotting = (PVXmlParamComboBox *) getParamWidgetByName("plotting");
        PVXmlParamComboBox *myMapping = (PVXmlParamComboBox *) getParamWidgetByName("mapping");
        PVXmlParamWidgetEditorBox *myName = (PVXmlParamWidgetEditorBox *) getParamWidgetByName("name");

        myMapping->clear();
        myMapping->addItem("");
        myMapping->addItems(getListTypeMapping(myType->currentText()));
        myMapping->select("default");

        myPlotting->clear();
        myPlotting->addItem("");
        myPlotting->addItems(getListTypePlotting(myType->currentText()));
        myPlotting->select("default");

        if (myType->currentText() == "time") {
            PVXmlParamTextEdit *timeFormat = (PVXmlParamTextEdit*) getParamWidgetByName("time-format");
            timeFormat->setVisible(true);
            myMapping->select("24h");
            myName->setText("Time");
        } else if (myType->currentText() == "integer") {
            myPlotting->select("minmax");
        } else {
            PVXmlParamTextEdit *timeFormat = (PVXmlParamTextEdit*) getParamWidgetByName("time-format");
            timeFormat->setVisible(false);
        }
    }
}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::slotConfirmRegExpInName
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::slotConfirmRegExpInName(const QString &name) {
    //for all char
    QRegExp reg(".*(\\*|\\[|\\{|\\]|\\}).*");
    if (reg.exactMatch(name)) {
        QDialog confirm(this);
        QVBoxLayout vb;
        confirm.setLayout(&vb);
        vb.addWidget(new QLabel("Are you writing a regular expression into the wrong field?"));
        QHBoxLayout bas;
        vb.addLayout(&bas);
        QPushButton no("No");
        bas.addWidget(&no);
        QPushButton yes("Yes");
        bas.addWidget(&yes);

        //connect the response button
        connect(&no, SIGNAL(clicked()), &confirm, SLOT(reject()));
        connect(&yes, SIGNAL(clicked()), &confirm, SLOT(accept()));

        //if confirmed then apply
        if (confirm.exec()) {
//            PVXmlParamTextEdit *name = (PVXmlParamTextEdit *) getParamWidgetByName("name");
//            PVXmlParamTextEdit *exp = (PVXmlParamTextEdit *) getParamWidgetByName("expression");
            //exp->setVal(name->getVal().toString());
            //name->setVal(QString(""));
        }
    }
}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::slotEmitNeedApply
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::slotEmitNeedApply() {
    emit signalNeedApply();
}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::slotSelectNext
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::slotSelectNext(){
  emit signalSelectNext();
}



