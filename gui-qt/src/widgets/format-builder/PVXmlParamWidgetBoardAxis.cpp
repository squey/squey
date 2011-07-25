///! \file PVXmlParamWidgetBoardAxis.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011
#include <PVXmlParamWidgetBoardAxis.h>
#include <picviz/PVMappingFilter.h>
#include <picviz/PVPlottingFilter.h>


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::PVXmlParamWidgetBoardAxis
 *
 *****************************************************************************/
PVInspector::PVXmlParamWidgetBoardAxis::PVXmlParamWidgetBoardAxis( PVRush::PVXmlTreeNodeDom *pNode):QWidget() {
    node = pNode;
    pluginListURL = picviz_plugins_get_functions_dir();
    setObjectName("PVXmlParamWidgetBoardAxis");

    allocBoardFields();
    draw();
    initValue();
    //updatePlotMapping(mapPlotType->val().toString());

    
    initConnexion();
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::~PVXmlParamWidgetBoardAxis
 *
 *****************************************************************************/
PVInspector::PVXmlParamWidgetBoardAxis::~PVXmlParamWidgetBoardAxis() {
    disableConnexion();
    hide();
    disAllocBoardFields();
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::allocBoardFields
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardAxis::allocBoardFields(){
    //tablWidget
    tabParam = new QTabWidget(this);
  
    //tab general
    //name
    textName = new PVXmlParamWidgetEditorBox(QString("name"), new QVariant(node->attribute("name")));
    //type
    mapPlotType = new PVXmlParamComboBox("type");
    timeFormatLabel = new QLabel("time format");
    timeFormat = new PVXmlParamTextEdit(QString("time-format"), QVariant(node->attribute("time-format")));    
    timeFormatStr = node->attribute("time-format");
    comboMapping = new PVXmlParamComboBox("mapping");
    comboPlotting = new PVXmlParamComboBox("plotting");
    
    //tab time format
    timeFormatInTab = new PVXmlParamTextEdit(QString("time-format"), QVariant(node->attribute("time-format")));  
    //validator
    timeSample = new PVXmlParamTextEdit(QString("time-sample"), QVariant(node->attribute("time-sample")));
    //html content for help
    helpTimeFormat = new QTextEdit();
    setHelp();
    //useParentRegExpValue = new QCheckBox("Do you want to use the value,\n from the parent regexp validator ?",this);
    
    //tab parameter
    comboKey = new PVXmlParamComboBox("key");
    keyLabel = new QLabel("key");
    group = new PVXmlParamWidgetEditorBox(QString("group"), new QVariant(node->attribute("group")));
    groupLabel = new QLabel("group");
    buttonColor = new PVXmlParamColorDialog("color", "#ffffff", this);
    colorLabel = new QLabel("color");
    buttonTitleColor = new PVXmlParamColorDialog("titlecolor", "#ffffff", this);
    titleColorLabel = new QLabel("titlecolor");
    //slotSetVisibleExtra(false);

    
    //button next
    buttonNextAxis = new QPushButton("Next");
}




/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::createTab
 *
 *****************************************************************************/
QVBoxLayout * PVInspector::PVXmlParamWidgetBoardAxis::createTab(const QString &title, QTabWidget *tab){
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
 * PVInspector::PVXmlParamWidgetBoardAxis::disAllocBoardFields
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardAxis::disAllocBoardFields(){
    //tab widget
    tabParam->hide();
    tabParam->deleteLater();
    
    //name
    textName->hide();
    textName->deleteLater();
    
    //tab general
    //type
    mapPlotType->hide();
    mapPlotType->deleteLater();
    comboMapping->hide();
    comboMapping->deleteLater();
    comboPlotting->hide();
    comboPlotting->deleteLater();
    timeFormatLabel->hide();
    timeFormatLabel->deleteLater();
    timeFormat->hide();
    timeFormat->deleteLater();
    
    //time format
    timeSample->hide();
    timeSample->deleteLater();
    helpTimeFormat->hide();
    helpTimeFormat->deleteLater();

    //extra group
    comboKey->hide();
    comboKey->deleteLater();
    comboKey->hide();
    group->deleteLater();
    buttonColor->hide();
    buttonColor->deleteLater();
    buttonTitleColor->hide();
    buttonTitleColor->deleteLater(); 
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::disableConnexion
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardAxis::disableConnexion(){
    disconnect(mapPlotType, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(updatePlotMapping(const QString&)));
    disconnect(textName, SIGNAL(textChanged(const QString&)), this, SLOT(slotSetValues()));
    disconnect(mapPlotType, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(slotSetValues()));
    disconnect(comboMapping, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(slotSetValues()));
    disconnect(comboPlotting, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(slotSetValues()));
    disconnect(timeFormat, SIGNAL(textChanged()), this, SLOT(slotSetValues()));
    disconnect(timeSample, SIGNAL(textChanged()), this, SLOT(slotSetValues()));
    disconnect(comboKey, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(slotSetValues()));
    disconnect(group, SIGNAL(textChanged(const QString&)), this, SLOT(slotSetValues()));
    disconnect(buttonColor, SIGNAL(changed()), this, SLOT(slotSetValues()));
    disconnect(buttonTitleColor, SIGNAL(changed()), this, SLOT(slotSetValues()));
    disconnect(buttonNextAxis,SIGNAL(clicked()), this, SLOT( slotGoNextAxis()));
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::draw
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardAxis::draw(){
    
    //alloc
    QVBoxLayout *layoutParam=new QVBoxLayout();
    //QVBoxLayout *layoutValues=new QVBoxLayout();
    QVBoxLayout *tabGeneral = createTab("General",tabParam);
    QVBoxLayout *tabParameter = createTab("Parameter",tabParam);
    QVBoxLayout *tabTimeFormat = createTab("Time Format",tabParam);
    QWidget *widgetTabAndNext = new QWidget(this);
    //QWidget *widgetValues = new QWidget(this);
    QHBoxLayout *layoutRoot = new QHBoxLayout(this);
    
    //QVBoxLayout *tabValuesApplied = createTab("Values applied",tabParam);
    
    //general layout
    //setLayout(layoutParam);
    setLayout(layoutRoot);
    layoutRoot->setContentsMargins(0,0,0,0);
    //tab widget
    layoutRoot->addWidget(widgetTabAndNext);
    layoutParam->setContentsMargins(0,0,0,0);
    widgetTabAndNext->setLayout(layoutParam);
//    layoutRoot->addWidget(widgetValues);
//    layoutValues->setContentsMargins(0,0,0,0);
//    widgetValues->setLayout(layoutValues);
    
    layoutParam->addWidget(tabParam);
    
    
    //***** tab general *****
    //name
    tabGeneral->addWidget(new QLabel("Axis name"));
    tabGeneral->addWidget(textName);
    //type
    tabGeneral->addWidget(new QLabel("type"));
    tabGeneral->addWidget(mapPlotType);
    //time edition
    tabGeneral->addWidget(timeFormatLabel);
    tabGeneral->addWidget(timeFormat);
    // Mapping/Plotting
    tabGeneral->addWidget(new QLabel("mapping"));
    tabGeneral->addWidget(comboMapping);
    tabGeneral->addWidget(new QLabel("plotting"));
    tabGeneral->addWidget(comboPlotting);
    tabGeneral->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Expanding, QSizePolicy::Expanding));
    
    //***** tab Time Format *****
    //time edition
    tabTimeFormat->addWidget(new QLabel("time format"));
    tabTimeFormat->addWidget(timeFormatInTab);
    tabTimeFormat->addWidget(helpTimeFormat);
    tabTimeFormat->addWidget(new QLabel("validator"));
    tabTimeFormat->addWidget(timeSample);
    
    
    //***** tab parameter *****
    tabParameter->addWidget(keyLabel);
    tabParameter->addWidget(comboKey);
    tabParameter->addWidget(groupLabel);
    tabParameter->addWidget(group);
    tabParameter->addWidget(colorLabel);
    tabParameter->addWidget(buttonColor);
    tabParameter->addWidget(titleColorLabel);
    tabParameter->addWidget(buttonTitleColor);
    tabParameter->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Expanding, QSizePolicy::Expanding));
    
    //***** view values from parent regexp *****
//    layoutValues->addWidget(new QLabel("Values form parent regexp.\n(click on RegExp if it's empty\nand be sure that there is\na text validator for the regexp.)"));
//    layoutValues->addWidget(tableValueFromParentRegExp);

    //button next
    layoutParam->addWidget(buttonNextAxis);
    //buttonNextAxis->setShortcut(QKeySequence(Qt::Key_Enter));
    buttonNextAxis->setShortcut(QKeySequence(Qt::Key_Return));
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::initConnexion
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardAxis::initConnexion() {
  
    connect(mapPlotType, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(updatePlotMapping(const QString&)));
    connect(textName, SIGNAL(textChanged(const QString&)), this, SLOT(slotSetValues()));
    connect(mapPlotType, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(slotSetValues()));
    connect(comboMapping, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(slotSetValues()));
    connect(comboPlotting, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(slotSetValues()));
    
    //time format
    //connect(timeFormat, SIGNAL(textChanged()), this, SLOT(slotSetValues()));
    connect(timeFormat, SIGNAL(textChanged()), this, SLOT(updateDateValidation()));
    //connect(timeFormatInTab, SIGNAL(textChanged()), this, SLOT(slotSetValues()));
    connect(timeFormatInTab, SIGNAL(textChanged()), this, SLOT(updateDateValidation()));
    connect(timeSample, SIGNAL(textChanged()), this, SLOT(slotSetValues()));
    
    
    //extra
    connect(comboKey, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(slotSetValues()));
    connect(group, SIGNAL(textChanged(const QString&)), this, SLOT(slotSetValues()));
    connect(buttonColor, SIGNAL(changed()), this, SLOT(slotSetValues()));
    connect(buttonTitleColor, SIGNAL(changed()), this, SLOT(slotSetValues()));
    
    //button next axis
    connect(buttonNextAxis,SIGNAL(clicked()), this, SLOT( slotGoNextAxis()));
    //buttonNextAxis->setShortCut();
}



/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::initValue
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardAxis::initValue(){
        QDir dir(pluginListURL);
    //init of combos
    QStringList typeL = QStringList(listType(dir.entryList()));
    mapPlotType->addItem("");
    mapPlotType->addItems(typeL);
    comboKey->addItem("true");
    comboKey->addItem("false");
    
    
    //type ...  auto select and default value
    if (node->attribute("type").length() > 1) {
        mapPlotType->select(node->attribute("type"));
        updatePlotMapping(node->attribute("type"));
    } else {
        mapPlotType->select("enum");
        updatePlotMapping("enum");
	slotSetVisibleTimeValid(false);
    }
    if (node->attribute("mapping").length() > 1)comboMapping->select(node->attribute("mapping"));
    else comboMapping->select("default");
    if (node->attribute("plotting").length() > 1)comboPlotting->select(node->attribute("plotting"));
    else comboPlotting->select("default");
    
    
    //extra
    if (node->attribute("key").length() > 1)comboKey->select(node->attribute("key"));
    else comboKey->select("false");
    if (node->attribute("color").length() > 1)buttonColor->setColor(node->attribute("color"));
    if (node->attribute("titlecolor").length() > 1)buttonTitleColor->setColor(node->attribute("titlecolor"));
    if (node->attribute("group").length() <= 1)group->setText("none");
    

}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardAxis::getWidgetToFocus
 *
 *****************************************************************************/
QWidget *PVInspector::PVXmlParamWidgetBoardAxis::getWidgetToFocus(){
  return (QWidget *)textName;
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::listType
 *
 *****************************************************************************/
QStringList PVInspector::PVXmlParamWidgetBoardAxis::listType(const QStringList &listEntry)const {
	LIB_FILTER(Picviz::PVMappingFilter)::list_filters const& map_filters = LIB_FILTER(Picviz::PVMappingFilter)::get().get_list();
	LIB_FILTER(Picviz::PVMappingFilter)::list_filters::const_iterator it;
	QStringList ret;
	for (it = map_filters.begin(); it != map_filters.end(); it++) {
		QString const& name = it.key();
		QStringList params = name.split('_');
		if (!ret.contains(params[0])) {
			ret << params[0];
		}
	}
    return ret;

}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::getListTypeMapping
 *
 *****************************************************************************/
QStringList PVInspector::PVXmlParamWidgetBoardAxis::getListTypeMapping(const QString& mType) {
	LIB_FILTER(Picviz::PVMappingFilter)::list_filters const& map_filters = LIB_FILTER(Picviz::PVMappingFilter)::get().get_list();
	LIB_FILTER(Picviz::PVMappingFilter)::list_filters::const_iterator it;
	QStringList ret;
	for (it = map_filters.begin(); it != map_filters.end(); it++) {
		QString const& name = it.key();
		QStringList params = name.split('_');
		if (params[0].compare(mType) == 0) {
			ret << params[1];
		}
	}
    return ret;
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::getListTypePlotting
 *
 *****************************************************************************/
QStringList PVInspector::PVXmlParamWidgetBoardAxis::getListTypePlotting(const QString& mType) {
	LIB_FILTER(Picviz::PVPlottingFilter)::list_filters const& pl_filters = LIB_FILTER(Picviz::PVPlottingFilter)::get().get_list();
	LIB_FILTER(Picviz::PVPlottingFilter)::list_filters::const_iterator it;
	QStringList ret;
	for (it = pl_filters.begin(); it != pl_filters.end(); it++) {
		QString const& name = it.key();
		QStringList params = name.split('_');
		if (params[0].compare(mType) == 0) {
			ret << params[1];
		}
	}
    return ret;
}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::setHelp
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardAxis::setHelp(){
  helpTimeFormat->setReadOnly(true);
  helpTimeFormat->document()->setDefaultStyleSheet("td {\nbackground-color:#ffe6bb;\n}\nbody{\nbackground-color:#fcffc4;\n}\n");
  QString html=QString("<body>\
  <big><b>HELP</b></big><br/>\
  sample :<br/>\
  MMM/d/yyyy H:m:s<br/>date\
  <table>\
  <tr><td>d</td><td>the day in month as a number (1 to 31)</td></tr>\
  <tr><td>e</td><td>the day of week as a number (1 to 31)</td></tr>\
  <tr><td>eee</td><td>the day of week as an abbreviated or long localized name (e.g. 'Mon' to 'Sun') or as a number (1 to 7)</td></tr>\
  <tr><td>eeee</td><td>the day of week as a long localized name (e.g. 'Monday' to 'Sunday') or as a number (1 to 7)</td></tr>\
  <tr><td>EEE</td><td>the day of week as an abbreviated or long localized name (e.g. 'Mon' to 'Sun')</td></tr>\
  <tr><td>EEEE</td><td>the day of week as long localized name (e.g. 'Monday' to 'Sunday')</td></tr>\
  <tr><td>M</td><td>the month as number with a leading zero (01-12)</td></tr>\
  <tr><td>MMM</td><td>the abbreviated or long localized month name (e.g. 'Jan' to 'Dec').</td></tr>\
  <tr><td>MMMM</td><td>the long localized month name (e.g. 'January' to 'December').</td></tr>\
  <tr><td>yy</td><td>the year as two digit number (00-99)</td></tr>\
  <tr><td>yyyy</td><td>the year as four digit number</td></tr>\
  </table>\
  <br /><strong>Note:</strong>&nbsp;the locale used for days and months in the log files is automatically found\
  <br/>Hour:\
  <table>\
  <tr> 		<td>h</td>		<td>hour in am/pm (00 to 12)</td>	</tr>\
  <tr> 		<td>H</td>		<td>hour in day (00 to 23)</td>	</tr>\
  <tr> 		<td>m</td>		<td>minute in hour (00 to 59)</td>	</tr>\
  <tr> 		<td>ss</td>		<td>second in minute  (00 to 59)</td>	</tr>\
  <tr> 		<td>S</td>		<td>fractional second (0 to 999)</td>	</tr>\
  <tr> 		<td>a</td>		<td>AM/PM marker</td>	</tr>\
  </table>\
  <br />Time zone:\
  <table>\
  <tr>		<td>Z</td>		<td>Time zone (RFC 822) (e.g. -0800)</td>	</tr>\
  <tr>		<td>v</td>		<td>Time zone (generic) (e.g. Pacific Time)</td>	</tr>\
  <tr>		<td>V</td>		<td>Time zone (abbreviation) (e.g. PT)</td>	</tr>\
  <tr>		<td>VVVV</td>	<td>Time zone (location) (e.g. United States (Los Angeles))</td>	</tr>\
  </table>\
  <br /><strong>Note:</strong>&nbsp;Any text that should be in the time format but not treated as special characters must be inside quotes (e.g. m'mn' s's')\
  </body>");
  
  //helpTimeFormat->setMi
  helpTimeFormat->setHtml(html);
}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::slotGoNextAxis
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardAxis::slotGoNextAxis(){
  if(!node->isOnRoot){//if we are not on root...
    emit signalSelectNext();
  }
}

/******************************************************************************
 *
 * VInspector::PVXmlParamWidgetBoardAxis::slotSetValues
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardAxis::slotSetValues(){

  //apply modification
    node->setAttribute(QString("name"),textName->text());
    node->setAttribute(QString("type"),mapPlotType->val().toString());
    node->setAttribute(QString("mapping"),comboMapping->val().toString());
    node->setAttribute(QString("plotting"),comboPlotting->val().toString());
    node->setAttribute(QString("time-format"),timeFormat->getVal().toString());
    node->setAttribute(QString("time-sample"),timeSample->getVal().toString());
    node->setAttribute(QString("key"),comboKey->val().toString());
    node->setAttribute(QString("group"),group->val().toString());
    node->setAttribute(QString("color"),buttonColor->getColor());
    node->setAttribute(QString("titlecolor"),buttonTitleColor->getColor());
   
    emit signalRefreshView();
}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::updateDateValidation
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardAxis::updateDateValidation(){
  if(timeSample->typeOfTextEdit==PVXmlParamTextEdit::dateValid){
    QString newText;//just declare
    if(timeFormatStr!=timeFormat->getVal().toString()){//timeFormat has changed...
      //get the new text
      newText = timeFormat->getVal().toString();
      timeFormatStr = newText;//memorise current value
      //modify timeFormatInTab
      timeFormatInTab->setVal(newText);//copy  current value in
    }else if(timeFormatStr!=timeFormatInTab->getVal().toString()){//timeFormat has changed...
      //get the new text
      newText = timeFormatInTab->getVal().toString();
      timeFormatStr = newText;
      //modify timeFormat
      timeFormat->setVal(timeFormatInTab->getVal().toString());
    }else{//no change
      return; 
    }
    //apply format
    timeSample->validDateFormat(newText.split("\n"));
    slotSetValues();
  }
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::slotSetVisibleTimeValid
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardAxis::slotSetVisibleTimeValid(bool flag){
  tabParam->setTabEnabled( 2, flag );
  timeFormatLabel->setVisible(flag);
  timeFormat->setVisible(flag);
}




/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::updatePlotMapping
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardAxis::updatePlotMapping(const QString& t) {
    //qDebug() << "updatePlotMapping(" << t << ")";
    if (t.length() > 1) {
        comboMapping->clear();
        comboMapping->addItem("");
        comboMapping->addItems(getListTypeMapping(mapPlotType->currentText()));
        comboMapping->select("default");

        comboPlotting->clear();
        comboPlotting->addItem("");
        comboPlotting->addItems(getListTypePlotting(mapPlotType->currentText()));
        comboPlotting->select("default");

        if (mapPlotType->currentText() == "time") {//if the type is a Time
	  //don't show time format editor
	    slotSetVisibleTimeValid(true);
	    //default selection
            comboMapping->select("24h");
	    //set the name with "Time" by default
	    if(node->attribute("name","Time").count()>1){
		textName->setText(node->attribute("name","Time"));
	    }else{
	        textName->setText("Time");
	    }
	    timeSample->validDateFormat(timeFormat->getVal().toString().split("\n"));
        } else if (mapPlotType->currentText() == "integer") {//if the type is an integer
            comboPlotting->select("minmax");
	    //don't show time format editor
	    slotSetVisibleTimeValid(false);
        } else {//if the type is not Time and not integer
	  //don't show time format editor
	    slotSetVisibleTimeValid(false);
        }
    }
}





