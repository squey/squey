///! \file PVXmlParamWidgetBoardAxis.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011
#include <picviz/PVMappingFilter.h>
#include <picviz/PVPlottingFilter.h>

#include <PVXmlParamWidgetBoardAxis.h>
#include <PVFormatBuilderWidget.h>
#include <PVAxisTagHelp.h>
#include <pvkernel/widgets/PVArgumentListWidget.h>
#include <picviz/widgets/PVArgumentListWidgetFactory.h>

#include <QDialogButtonBox>

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::PVXmlParamWidgetBoardAxis
 *
 *****************************************************************************/
PVInspector::PVXmlParamWidgetBoardAxis::PVXmlParamWidgetBoardAxis(PVRush::PVXmlTreeNodeDom *pNode, PVXmlParamWidget* parent):
	QWidget(),
	_parent(parent)
{
    node = pNode;
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
    mapPlotType = new PVWidgetsHelpers::PVAxisTypeWidget(this);
    comboMapping = new PVWidgetsHelpers::PVMappingModeWidget(this);
    comboPlotting = new PVWidgetsHelpers::PVPlottingModeWidget(this);
    
    //tab parameter
    group = new PVXmlParamWidgetEditorBox(QString("group"), new QVariant(node->attribute(PVFORMAT_AXIS_GROUP_STR)));
    groupLabel = new QLabel(tr("Group :"));
	comboGroup = new PVXmlParamComboBox("group");
	btnGroupAdd = new QPushButton(tr("Add a group..."));
	listTags = new PVXmlParamList("tag");
	btnTagHelp = new QPushButton(QIcon(":/help"), "Help");
    buttonColor = new PVXmlParamColorDialog("color", PVFORMAT_AXIS_COLOR_DEFAULT, this);
    colorLabel = new QLabel(tr("Color of the axis line :"));
    buttonTitleColor = new PVXmlParamColorDialog("titlecolor", PVFORMAT_AXIS_TITLECOLOR_DEFAULT, this);
    titleColorLabel = new QLabel(tr("Color of the axis title :"));
    //slotSetVisibleExtra(false);
	
	_layout_params_mp = new QHBoxLayout();
	_params_mapping = new PVWidgets::PVArgumentListWidget(PVWidgets::PVArgumentListWidgetFactory::create_mapping_plotting_widget_factory(), this);
	_params_plotting = new PVWidgets::PVArgumentListWidget(PVWidgets::PVArgumentListWidgetFactory::create_mapping_plotting_widget_factory(), this);

	QSizePolicy grp_params_policy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	grp_params_policy.setHorizontalStretch(1);
	_grp_mapping = new QGroupBox(tr("Mapping properties"));
	_grp_plotting = new QGroupBox(tr("Plotting properties"));
	_params_mapping->setSizePolicy(grp_params_policy);
	_params_mapping->setSizePolicy(grp_params_policy);
	_params_plotting->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	QVBoxLayout* tmp_layout = new QVBoxLayout();
	tmp_layout->addWidget(_params_mapping);
	_grp_mapping->setLayout(tmp_layout);
	tmp_layout = new QVBoxLayout();
	tmp_layout->addWidget(_params_plotting);
	_grp_plotting->setLayout(tmp_layout);
    
    //button next
    buttonNextAxis = new QPushButton(tr("Next"));
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
    
    //extra group
    group->deleteLater();
	comboGroup->deleteLater();
	btnGroupAdd->deleteLater();
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
    disconnect(listTags, SIGNAL(itemSelectionChanged()), this, SLOT(slotSetValues()));
    disconnect(group, SIGNAL(textChanged(const QString&)), this, SLOT(slotSetValues()));
	disconnect(btnGroupAdd, SIGNAL(clicked()), this, SLOT(slotAddGroup()));
    disconnect(buttonColor, SIGNAL(changed()), this, SLOT(slotSetValues()));
    disconnect(buttonTitleColor, SIGNAL(changed()), this, SLOT(slotSetValues()));
    disconnect(buttonNextAxis,SIGNAL(clicked()), this, SLOT( slotGoNextAxis()));
	disconnect(btnTagHelp, SIGNAL(clicked()), this, SLOT(slotShowTagHelp()));
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
	QGridLayout* gridLayout = new QGridLayout();
	int i = 0;
    //name
    gridLayout->addWidget(new QLabel(tr("Axis name :")), i, 0);
    gridLayout->addWidget(textName, i, 2, 1, -1);
	i += 2;
	// tag
	gridLayout->addWidget(new QLabel(tr("Tag :")), i, 0);
	gridLayout->addWidget(listTags, i, 2);
	gridLayout->addWidget(btnTagHelp, i, 4);
	i += 2;
    //type
    gridLayout->addWidget(new QLabel(tr("Type :")), i, 0);
    gridLayout->addWidget(mapPlotType, i, 2, 1, -1);
	i += 2;
    // Mapping/Plotting
    gridLayout->addWidget(new QLabel(tr("Mapping :")), i, 0);
    gridLayout->addWidget(comboMapping, i, 2, 1, -1);
	i += 2;
    gridLayout->addWidget(new QLabel(tr("Plotting :")), i, 0);
    gridLayout->addWidget(comboPlotting, i, 2, 1, -1);
	tabGeneral->addLayout(gridLayout);
	// Mapping/plotting properties
	_layout_params_mp->addWidget(_grp_mapping);
	_layout_params_mp->addWidget(_grp_plotting);
	tabGeneral->addLayout(_layout_params_mp);
    tabGeneral->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Expanding, QSizePolicy::Expanding));
    
    //***** tab Time Format *****
    //***** tab parameter *****
	gridLayout = new QGridLayout();
	i = 0;
    gridLayout->addWidget(groupLabel, i, 0);
	gridLayout->addWidget(comboGroup, i, 2);
	gridLayout->addWidget(btnGroupAdd, i, 4);
	i += 2;
    gridLayout->addWidget(colorLabel, i, 0);
    gridLayout->addWidget(buttonColor, i, 2, 1, -1);
	i += 2;
    gridLayout->addWidget(titleColorLabel, i, 0);
    gridLayout->addWidget(buttonTitleColor, i, 2, 1, -1);
	tabParameter->addLayout(gridLayout);
    tabParameter->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Expanding, QSizePolicy::Expanding));
    
    //***** view values from parent regexp *****
//    layoutValues->addWidget(new QLabel("Values form parent regexp.\n(click on RegExp if it's empty\nand be sure that there is\na text validator for the regexp.)"));
//    layoutValues->addWidget(tableValueFromParentRegExp);

    //button next
    layoutParam->addWidget(buttonNextAxis);
    //buttonNextAxis->setShortcut(QKeySequence(Qt::Key_Enter));
    buttonNextAxis->setShortcut(QKeySequence(Qt::Key_Return));
	
	checkMappingTimeFormat();
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
    connect(comboMapping, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(updateMappingParams()));
	connect(_params_mapping, SIGNAL(args_changed_Signal()), this, SLOT(slotSetParamsMapping()));
    connect(comboPlotting, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(slotSetValues()));
    connect(comboPlotting, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(updatePlottingParams()));
	connect(_params_plotting, SIGNAL(args_changed_Signal()), this, SLOT(slotSetParamsPlotting()));
    connect(listTags, SIGNAL(itemSelectionChanged()), this, SLOT(slotSetValues()));
	connect(btnTagHelp, SIGNAL(clicked()), this, SLOT(slotShowTagHelp()));
    
    //extra
    connect(group, SIGNAL(textChanged(const QString&)), this, SLOT(slotSetValues()));
	connect(btnGroupAdd, SIGNAL(clicked()), this, SLOT(slotAddGroup()));
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
void PVInspector::PVXmlParamWidgetBoardAxis::initValue()
{
    //init of combos
    
    //type ...  auto select and default value
	QString node_type = node->attribute(PVFORMAT_AXIS_TYPE_STR);
    if (node_type.isEmpty()) {
		node_type = PVFORMAT_AXIS_TYPE_DEFAULT;
    }
	mapPlotType->sel_type(node_type);
	updatePlotMapping(node_type);

	_args_mapping.clear();
	Picviz::PVMappingFilter::p_type map_lib = get_mapping_lib_filter();
	QString node_mapping = node->getMappingProperties(map_lib->get_default_args(), _args_mapping);
    if (node_mapping.isEmpty()) {
		node_mapping = PVFORMAT_AXIS_MAPPING_DEFAULT;
	}
	comboMapping->set_mode(node_mapping);
	_args_map_mode[*map_lib] = _args_mapping;
	_params_mapping->set_args(_args_mapping);
	
	_args_plotting.clear();
	Picviz::PVPlottingFilter::p_type plot_lib = get_plotting_lib_filter();
	QString node_plotting = node->getPlottingProperties(plot_lib->get_default_args(), _args_plotting);
    if (node_plotting.isEmpty()) {
		node_plotting = PVFORMAT_AXIS_PLOTTING_DEFAULT;
	}
	comboPlotting->set_mode(node_plotting);
	_args_plot_mode[*plot_lib] = _args_plotting;
	_params_plotting->set_args(_args_plotting);
    
    
    //extra
	QString node_color = node->attribute(PVFORMAT_AXIS_COLOR_STR);
    if (node_color.isEmpty()) {
		node_color = PVFORMAT_AXIS_COLOR_DEFAULT;
	}
	buttonColor->setColor(node_color);

	QString node_tc = node->attribute(PVFORMAT_AXIS_TITLECOLOR_STR);
    if (node_tc.isEmpty()) {
		node_tc = PVFORMAT_AXIS_TITLECOLOR_DEFAULT;
	}
	buttonTitleColor->setColor(node_tc);

	setComboGroup();
	setListTags();
	//updateMappingParams();
	//updatePlottingParams();
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
QStringList PVInspector::PVXmlParamWidgetBoardAxis::listType() const {
	return Picviz::PVMappingFilter::list_types();
}


/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::getListTypeMapping
 *
 *****************************************************************************/
QStringList PVInspector::PVXmlParamWidgetBoardAxis::getListTypeMapping(const QString& mType) {
	return Picviz::PVMappingFilter::list_modes(mType);
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::getListTypePlotting
 *
 *****************************************************************************/
QStringList PVInspector::PVXmlParamWidgetBoardAxis::getListTypePlotting(const QString& mType) {
	return Picviz::PVPlottingFilter::list_modes(mType);
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
    node->setAttribute(QString(PVFORMAT_AXIS_NAME_STR),textName->text());
    node->setAttribute(QString(PVFORMAT_AXIS_TYPE_STR),mapPlotType->get_sel_type());
    node->setAttribute(QString(PVFORMAT_AXIS_GROUP_STR),group->val().toString());
    node->setAttribute(QString(PVFORMAT_AXIS_COLOR_STR),buttonColor->getColor());
    node->setAttribute(QString(PVFORMAT_AXIS_TITLECOLOR_STR),buttonTitleColor->getColor());
    node->setAttribute(QString(PVFORMAT_AXIS_TAG_STR),listTags->selectedList().join(QString(QChar(PVFORMAT_TAGS_SEP))));

	updateMappingParams();
	updatePlottingParams();
   
    emit signalRefreshView();
}

void PVInspector::PVXmlParamWidgetBoardAxis::slotSetParamsMapping()
{
	QString mode = comboMapping->get_mode();
	Picviz::PVMappingFilter::p_type lib_filter = get_mapping_lib_filter();
	if (!lib_filter) {
		return;
	}
	_args_map_mode[*lib_filter] = _args_mapping;
	node->setMappingProperties(mode, lib_filter->get_default_args(), _args_mapping);
}

void PVInspector::PVXmlParamWidgetBoardAxis::slotSetParamsPlotting()
{
	QString mode = comboPlotting->get_mode();
	Picviz::PVPlottingFilter::p_type lib_filter = get_plotting_lib_filter();
	if (!lib_filter) {
		return;
	}
	_args_plot_mode[*lib_filter] = _args_plotting;
	node->setPlottingProperties(mode, lib_filter->get_default_args(), _args_plotting);
}


void PVInspector::PVXmlParamWidgetBoardAxis::checkMappingTimeFormat()
{
#if 0
	// AG: this is a hack. Check that, if type is "time", that, if a "week" or
	// "24h" mapping has been set, the good time-format comes with it.
	QString time_mapping = comboMapping->get_mode();
	comboMapping->clear_disabled_strings();

	// 24h mapping:  check that 'h' or 'H' are present
	bool reset_mapping = false;
	if (timeFormatStr.indexOf(QChar('h'), 0, Qt::CaseInsensitive) == -1) {
		comboMapping->add_disabled_string("24h");
		reset_mapping = (time_mapping == "24h");
	}
	if (timeFormatStr.indexOf(QChar('d'), 0, Qt::CaseInsensitive) == -1 &&
			timeFormatStr.indexOf(QChar('e'), 0, Qt::CaseInsensitive) == -1) {
		comboMapping->add_disabled_string("week");
		reset_mapping = (time_mapping == "week");
	}
	if (reset_mapping) {
		comboMapping->select_default();
	}
#endif
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::updatePlotMapping
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardAxis::updatePlotMapping(const QString& t)
{
	if (t.length() > 1) {
		QString type = mapPlotType->get_sel_type();
		comboMapping->clear();
		comboMapping->populate_from_type(type);
		comboMapping->select_default();

		comboPlotting->clear();
		comboPlotting->populate_from_type(mapPlotType->get_sel_type());
		comboPlotting->select_default();
	}
}

Picviz::PVMappingFilter::p_type PVInspector::PVXmlParamWidgetBoardAxis::get_mapping_lib_filter()
{
	Picviz::PVMappingFilter::p_type lib_filter;
	QString mode = comboMapping->get_mode();
	if (!mode.isEmpty()) {
		lib_filter = LIB_CLASS(Picviz::PVMappingFilter)::get().get_class_by_name(mapPlotType->get_sel_type() + "_" + mode);
	}

	return lib_filter;
}

Picviz::PVPlottingFilter::p_type PVInspector::PVXmlParamWidgetBoardAxis::get_plotting_lib_filter()
{
	Picviz::PVPlottingFilter::p_type lib_filter;
	QString mode = comboPlotting->get_mode();
	if (!mode.isEmpty()) {
		lib_filter = LIB_CLASS(Picviz::PVPlottingFilter)::get().get_class_by_name(mapPlotType->get_sel_type() + "_" + mode);
	}

	return lib_filter;
}

void PVInspector::PVXmlParamWidgetBoardAxis::updateMappingParams()
{
	_args_mapping.clear();
	Picviz::PVMappingFilter::p_type lib_filter = get_mapping_lib_filter();
	if (!lib_filter) {
		return;
	}
	std::map<Picviz::PVMappingFilter::base_registrable, PVCore::PVArgumentList>::iterator it;
	if ((it = _args_map_mode.find(*lib_filter)) != _args_map_mode.end()) {
		_args_mapping = it->second;
	}
	else {
		_args_mapping = lib_filter->get_default_args();
	}
	_params_mapping->set_args(_args_mapping);

	slotSetParamsMapping();
}

void PVInspector::PVXmlParamWidgetBoardAxis::updatePlottingParams()
{
	_args_plotting.clear();
	Picviz::PVPlottingFilter::p_type lib_filter = get_plotting_lib_filter();
	if (!lib_filter) {
		return;
	}
	std::map<Picviz::PVPlottingFilter::base_registrable, PVCore::PVArgumentList>::iterator it;
	if ((it = _args_plot_mode.find(*lib_filter)) != _args_plot_mode.end()) {
		_args_plotting = it->second;
	}
	else {
		_args_plotting = lib_filter->get_default_args();
	}
	_params_plotting->set_args(_args_plotting);

	slotSetParamsPlotting();
}

void PVInspector::PVXmlParamWidgetBoardAxis::setComboGroup()
{
	QString type = mapPlotType->get_sel_type();
	PVFormatBuilderWidget* editor = parent()->parent();
	PVRush::types_groups_t const& types_grps = editor->getGroups();
	if (!types_grps.contains(type)) {
		return;
	}

	comboGroup->clear();
	comboGroup->addItem(PVFORMAT_AXIS_GROUP_DEFAULT);
	QStringList grps = types_grps[type].toList();
	for (int i = 0; i < grps.size(); i++) {
		comboGroup->addItem(grps[i]);
	}

	QString node_group = node->attribute(PVFORMAT_AXIS_GROUP_STR);
    if (node_group.isEmpty()) {
		node_group = PVFORMAT_AXIS_GROUP_DEFAULT;
	}
	comboGroup->select(node_group);
}

void PVInspector::PVXmlParamWidgetBoardAxis::setListTags()
{
	listTags->clear();
	listTags->addItem(PVFORMAT_AXIS_TAG_DEFAULT);

	QSet<QString> list_tags = getListTags();
	QSet<QString> list_splitter_tags = getListParentSplitterTag();

	listTags->setItems(list_tags.unite(list_splitter_tags).toList());

	QString node_tag = node->attribute(PVFORMAT_AXIS_TAG_STR);
	if (node_tag.isEmpty()) {
		node_tag = PVFORMAT_AXIS_TAG_DEFAULT;
	}
	listTags->select(node_tag.split(PVFORMAT_TAGS_SEP));
}

void PVInspector::PVXmlParamWidgetBoardAxis::slotAddGroup()
{
	QString type = mapPlotType->get_sel_type();
	QDialog* add_dlg = new QDialog(parent()->parent());
	add_dlg->setWindowTitle(tr("Add a group..."));

	QVBoxLayout* mainLayout = new QVBoxLayout();
	QHBoxLayout* nameLayout = new QHBoxLayout();
	nameLayout->addWidget(new QLabel(tr("Group name") + QString(" :")));
	QLineEdit* group_name = new QLineEdit();
	nameLayout->addWidget(group_name);
	mainLayout->addLayout(nameLayout);
	mainLayout->addWidget(new QLabel(tr("That group will be added for the type %1.").arg(type)));
	QDialogButtonBox* btns = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	connect(btns, SIGNAL(accepted()), add_dlg, SLOT(accept()));
	connect(btns, SIGNAL(rejected()), add_dlg, SLOT(reject()));
	mainLayout->addWidget(btns);
	add_dlg->setLayout(mainLayout);

	if (add_dlg->exec() == QDialog::Rejected) {
		return;
	}

	QString new_grp = group_name->text();
	PVRush::types_groups_t& types_grps = parent()->parent()->getGroups();
	types_grps[type] << new_grp;

	node->setAttribute(PVFORMAT_AXIS_GROUP_STR, new_grp);
	setComboGroup();
}

QSet<QString> PVInspector::PVXmlParamWidgetBoardAxis::getListTags()
{
	QSet<QString> ret;
	Picviz::PVLayerFilterListTags const& lt = LIB_CLASS(Picviz::PVLayerFilter)::get().get_tags();
	Picviz::PVLayerFilterListTags::const_iterator it;
	for (it = lt.begin(); it != lt.end(); it++) {
		Picviz::PVLayerFilterTag const& tag = *it;
		ret << (QString) tag;
	}
	return ret;
}

QSet<QString> PVInspector::PVXmlParamWidgetBoardAxis::getListParentSplitterTag()
{
	QSet<QString> ret;
	PVRush::PVXmlTreeNodeDom* parent = node->getParent();
	if (!parent) {
		return ret;
	}
	parent = parent->getParent();
	if (!parent || parent->type != PVRush::PVXmlTreeNodeDom::splitter) {
		return ret;
	}

	// Ok, we have a splitter has parent. Let's get its provided tags.
	PVFilter::PVFieldsSplitter_p filter_p = LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_class_by_name(parent->attribute("type", ""));
	assert(filter_p);

	// Ok, get the tags !
	PVFilter::PVFieldsSplitterListTags const& tags = LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_tags_for_class(*filter_p);
	for (int i = 0; i < tags.size(); i++) {
		PVFilter::PVFieldsSplitterTag const& tag = tags.at(i);
		ret << (QString) tag;
	}

	return ret;
}

QStringList PVInspector::PVXmlParamWidgetBoardAxis::get_current_tags()
{
	return listTags->selectedList();
}

void PVInspector::PVXmlParamWidgetBoardAxis::slotShowTagHelp()
{
	PVAxisTagHelp* dlg = new PVAxisTagHelp(get_current_tags(), parent()->parent());
	dlg->exec();
}
