///! \file PVXmlParamWidgetBoardSplitterRegEx.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#include <PVXmlParamWidgetBoardSplitterRegEx.h>


#define dbg {qDebug()<<__FILE__<<":"<<__LINE__;}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardSplitterRegEx::PVXmlParamWidgetBoardSplitterRegEx
 *
 *****************************************************************************/
PVInspector::PVXmlParamWidgetBoardSplitterRegEx::PVXmlParamWidgetBoardSplitterRegEx(PVInspector::PVXmlTreeNodeDom *pNode) : QWidget() {
    node = pNode;
    allocBoardFields();
    draw();
    flagSaveRegExpValidator = false;
    initValue();
    initConnexion();
    validWidget->setRegEx(exp->text());
    flagNeedConfirmAndSave = false;
    flagAskConfirmActivated = true;
    
}



/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardSplitterRegEx::~PVXmlParamWidgetBoardSplitterRegEx
 *
 *****************************************************************************/
PVInspector::PVXmlParamWidgetBoardSplitterRegEx::~PVXmlParamWidgetBoardSplitterRegEx() {
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::allocBoardFields
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::allocBoardFields() {
    tabParam = new QTabWidget(this);
    
    //tab 
    name = new PVXmlParamWidgetEditorBox(QString("name"), new QVariant(node->getAttribute("name")));
    
    //tab regexp
    exp = new PVXmlParamWidgetEditorBox(QString("expression"), new QVariant(node->getDom().attribute("expression", ".*")));
    labelNbr = new QLabel("");
    openLog = new QPushButton("Open a log");
    checkSaveValidLog = new QCheckBox("Save log sample in format file",this);
    validWidget = new PVXmlParamTextEdit(QString("validator"), QVariant(node->getAttribute("validator",false)));
    table = new QTableWidget();
    btnApply = new QPushButton("Apply");

}

/******************************************************************************
 *
 * bool PVInspector::PVXmlParamWidgetBoardSplitterRegEx::confirmAndSave
 *
 *****************************************************************************/
bool PVInspector::PVXmlParamWidgetBoardSplitterRegEx::confirmAndSave() {
  //open the confirm box.
    QDialog confirm(this);
    QVBoxLayout vb;
    confirm.setLayout(&vb);
    vb.addWidget(new QLabel("Do you want to apply the modifications ?"));
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
    return confirm.exec();

}

/******************************************************************************
 *
 * VInspector::PVXmlParamWidgetBoardSplitterRegEx::createTab
 *
 *****************************************************************************/
QVBoxLayout * PVInspector::PVXmlParamWidgetBoardSplitterRegEx::createTab(const QString &title, QTabWidget *tab){
    QWidget *tabWidget = new QWidget(tab);
    //create the layout
    QVBoxLayout *tabWidgetLayout = new QVBoxLayout(tabWidget);
    
    //creation of the tab
    tabWidgetLayout->setContentsMargins(0,0,0,0);
    tabWidget->setLayout(tabWidgetLayout);
    
    //add the tab
    tab->addTab(tabWidget,title);
    
    //return the layout to add items
    return tabWidgetLayout;
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::disableConnexion
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::disableConnexion() {
    disconnect(name, SIGNAL(textChanged(const QString&)), this, SLOT(slotSetValues()));
    disconnect(exp, SIGNAL(textChanged(const QString&)), validWidget, SLOT(setRegEx(const QString &)));
    disconnect(exp, SIGNAL(textChanged(const QString&)), this, SLOT(slotSetValues()));
    disconnect(exp, SIGNAL(textChanged(const QString&)), this, SLOT(regExCount(const QString&)));
    disconnect(validWidget, SIGNAL(textChanged()), this, SLOT(slotSetValues()));
    disconnect(validWidget, SIGNAL(textChanged()), this, SLOT(slotNoteConfirmationNeeded()));
    disconnect(btnApply, SIGNAL(clicked()), this, SLOT(slotSetConfirmedValues()));
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::disAllocBoardFields
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::disAllocBoardFields() {

}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::draw
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::draw() {
  //init layout
    QVBoxLayout *qv = new QVBoxLayout();
    QVBoxLayout *tabReg = createTab("regexp",tabParam);
    QVBoxLayout *tabGeneral = createTab("general",tabParam);
    
    
    //init the parameter board layout
    qv->addWidget(tabParam);
    

    
    //tab general
    //field name
    tabGeneral->addWidget(new QLabel("RegEx name"));
    tabGeneral->addWidget(name);
    tabGeneral->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Expanding, QSizePolicy::Expanding));
    //field expression
   
    
    //tab regexp
    tabReg->addWidget(new QLabel("Expression"));
    tabReg->addWidget(exp);
    //exp->setFocus();
    tabReg->addWidget(labelNbr);
    tabReg->addWidget(openLog);
    tabReg->addWidget(checkSaveValidLog);
    tabReg->addWidget(validWidget);
    tabReg->addWidget(new QLabel("view of selection"));
    tabReg->addWidget(table);
    //apply
    tabReg->addWidget(btnApply);
    //btnApply->setShortcut(QKeySequence(Qt::Key_Enter|Qt::Key_Return));
    btnApply->setShortcut(QKeySequence(Qt::Key_Return));

    
    
    

    setLayout(qv);
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::exit
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::exit() {
  
  //open the confirmbox if we quit a regexp
    if (flagNeedConfirmAndSave && flagAskConfirmActivated) {
        if (confirmAndSave()) {
            slotSetConfirmedValues();
        }
        flagNeedConfirmAndSave = false;
	flagAskConfirmActivated = false;
    }
    
    disableConnexion();
    if(table->isVisible())table->setVisible(false);
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::initConnexion
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::initConnexion() {
    //tab general
    connect(name, SIGNAL(textChanged(const QString&)), this, SLOT(slotSetValues()));//to update tree view.
    connect(name, SIGNAL(textChanged(const QString&)), this, SLOT(slotVerifRegExpInName()));//to verify if regexp is write in name.
    //tab regexp
    connect(exp, SIGNAL(textChanged(const QString&)), validWidget, SLOT(setRegEx(const QString &)));//to update regexp
    connect(exp, SIGNAL(textChanged(const QString&)), this, SLOT(slotNoteConfirmationNeeded()));//to note that we need to confirm change
    connect(exp, SIGNAL(textChanged(const QString&)), this, SLOT(regExCount(const QString&)));//to update the numbre of field which are detected
    connect(validWidget, SIGNAL(textChanged()), this, SLOT(slotNoteConfirmationNeeded()));//to note that we need to confirm change
    connect(validWidget, SIGNAL(textChanged()), this, SLOT(slotUpdateTable()));//to update the text validator
    connect(checkSaveValidLog,SIGNAL(clicked(bool)),this,SLOT(slotSaveValidator(bool)));
    connect(checkSaveValidLog,SIGNAL(clicked(bool)),this,SLOT(slotSetConfirmedValues()));
    connect(openLog, SIGNAL(clicked()), this, SLOT(slotOpenLogValid()));//to choose a log for the validator
    connect(btnApply, SIGNAL(clicked()), this, SLOT(slotSetConfirmedValues()));//to apply modification.
}



/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::initValue
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::initValue() {
    //init the number of field detected with the regexp
    regExCount(exp->val().toString());
    //check or not the check box
    if(node->getAttribute("saveValidator","").compare(QString("true"))==0){
	flagSaveRegExpValidator=true;
	checkSaveValidLog->setCheckState(Qt::Checked);
	validWidget->setVal(node->getAttribute("validator",true));
    }else{
	flagSaveRegExpValidator=false;
	checkSaveValidLog->setCheckState(Qt::Unchecked);
	validWidget->setVal(node->getAttribute("validator",false));
    }
}


/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::getWidgetToFocus
 *
 *****************************************************************************/
QWidget *PVInspector::PVXmlParamWidgetBoardSplitterRegEx::getWidgetToFocus(){
  return (QWidget *)name;
}



/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::regExCount
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::regExCount(const QString &reg) {
    //set regexp
    QRegExp regExp = QRegExp(reg);
    nbr = regExp.captureCount();
    //set tne number of field
    labelNbr->setText(QString("selection count : %1").arg(nbr));

    if(nbr>=1){//if there is at least one selection...
	//get selection detection pettern.
	QString patternToDetectSel;
	for(int i=0;i<nbr;i++){
	  patternToDetectSel += QString(".*(\\([^)]*\\)).*");
	}
	QRegExp regDetectSel = QRegExp(QString(patternToDetectSel));
	//detect selections in regexp.
	regDetectSel.indexIn(reg,0);
	//set string list of each selection pattern
	for(int i=1;i<=nbr;i++){//for each selection...
	    node->setAttribute(QString("selectionRegExp-%0").arg(i),regDetectSel.cap(i).toUtf8().constData());
	}
    }
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotNoteConfirmationNeeded
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotNoteConfirmationNeeded() {
    flagNeedConfirmAndSave = true;//note that we need confirmation
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotOpenLogValid
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotOpenLogValid() {
    QFileDialog fd;
    QString urlFile = fd.getOpenFileName(0, QString("Select the file."), "."); //open file chooser
    QFile f(urlFile);
    if (f.exists()){//if the file is valid
        if(!f.open(QIODevice::ReadOnly | QIODevice::Text))return;
        for(int i=0;i<50;i++){//for the 50 first line ...
            QString l = validWidget->getVal().toString();
            l.push_back(QString(f.readLine()));//...add the line in validator
            validWidget->setVal(l);
        }
    }
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotSaveValidator
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotSaveValidator(bool stat){
    flagSaveRegExpValidator = stat;
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotSetConfirmedValues
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotSetConfirmedValues() {
    slotSetValues();//save various value
    node->setAttribute(QString("expression"), exp->text());//save expression
    node->setAttribute(QString("validator"), validWidget->getVal().toString(),flagSaveRegExpValidator);//save the text in validator

    regExCount(exp->text());
    node->setNbr(nbr);//set the fileds with expression rexexp selection count.
    flagNeedConfirmAndSave = false;
    emit signalRefreshView();
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotSetValues
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotSetValues() {
    node->setAttribute(QString("name"), name->text());//save name
    if(flagSaveRegExpValidator){
      node->setAttribute(QString("saveValidator"),"true");
    }else{
      node->setAttribute(QString("saveValidator"),"false");
    }
    
    emit signalRefreshView();
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotShowTable
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotShowTable(bool isVisible) {
    table->setVisible(isVisible);//hide or show table validator selection
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotVerifRegExpInName
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotVerifRegExpInName() {
  //char we want to detecte in the name
    QRegExp reg(".*(\\*|\\[|\\{|\\]|\\}).*");
    if (reg.exactMatch(name->text())) {//if there is
      //create the confirm popup
        QDialog confirm(this);
        QVBoxLayout vb;
        confirm.setLayout(&vb);
        vb.addWidget(new QLabel("you are maybe writting a regexp in the name field. Do you want to write it in the expression field ?"));
        QHBoxLayout bas;
        vb.addLayout(&bas);
        QPushButton no("No");
        bas.addWidget(&no);
        QPushButton yes("Yes");
        bas.addWidget(&yes);

        //connect the response button
        connect(&no, SIGNAL(clicked()), &confirm, SLOT(reject()));
        connect(&yes, SIGNAL(clicked()), &confirm, SLOT(accept()));

        
        if (confirm.exec()) {//if confirmed then apply...
            exp->setText(name->text());//push text
            name->setText("");
        }
    }
}


/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotUpdateTable
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotUpdateTable() {
    QRegExp reg = QRegExp(exp->text());

    //update the number of column
    reg.indexIn(validWidget->getVal().toString(), 0);
    table->setColumnCount(reg.captureCount());


    //feed each line with the matching in text zone.
    QStringList myText = validWidget->getVal().toString().split("\n");
    table->setRowCount(myText.count());
    updateHeaderTable();
    for (int line = 0; line < myText.count(); line++) {//for each line...
        QString myLine = myText.at(line);
        if (reg.exactMatch(myLine)) {
            for (int cap = 0; cap < reg.captureCount(); cap++) {//for each column (regexp selection)...
                reg.indexIn(myLine, 0);
                table->setItem(line, cap, new QTableWidgetItem(reg.cap(cap + 1)));
                int width = 12 + (8 * reg.cap(cap + 1).length());
                if (width > table->columnWidth(cap)) {
		  table->setColumnWidth(cap, width); //update the size
		}
            }
        }
    }
    table->setContentsMargins(3, 0, 3, 0);
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::updateHeaderTable
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::updateHeaderTable(){
    QStringList l;
    for(int i=0;i<node->countChildren();i++){
        l.push_back(node->getChild(i)->getOutName());
        table->setColumnWidth(i, 1);
        int width = 12 + (8 * node->getChild(i)->getOutName().length());
        if (width > table->columnWidth(i)){
	  table->setColumnWidth(i, width); //update the size
	}
    }
    //qDebug()<<l;
    //table->setHorizontalHeaderItem(1,"e")
    table->setHorizontalHeaderLabels(l);
    table->setContentsMargins(3, 0, 3, 0);
    
}
