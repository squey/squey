/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVMappingFilter.h>
#include <inendi/PVPlottingFilter.h>

#include <PVXmlParamWidgetBoardAxis.h>
#include <PVFormatBuilderWidget.h>
#include <PVAxisTagHelp.h>
#include <pvkernel/widgets/PVArgumentListWidget.h>
#include <inendi/widgets/PVArgumentListWidgetFactory.h>
#include <pvkernel/widgets/editors/PVTimeFormatEditor.h>

#include <QDialogButtonBox>

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::PVXmlParamWidgetBoardAxis
 *
 *****************************************************************************/
PVInspector::PVXmlParamWidgetBoardAxis::PVXmlParamWidgetBoardAxis(PVRush::PVXmlTreeNodeDom* pNode,
                                                                  PVXmlParamWidget* parent)
    : QWidget(), _parent(parent)
{
	node = pNode;
	setObjectName("PVXmlParamWidgetBoardAxis");

	allocBoardFields();
	draw();
	initConnexion();
	initValue();
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::allocBoardFields
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardAxis::allocBoardFields()
{
	// tablWidget
	tabParam = new QTabWidget(this);

	// tab general
	// name
	textName =
	    new PVXmlParamWidgetEditorBox(QString("name"), new QVariant(node->attribute("name")));
	// Try to read type format from xml file as default value.

	_type_format = new PVXmlParamWidgetEditorBox(QString("type format"),
	                                             new QVariant(node->attribute("type_format")));
	btnTypeFormatHelp = new QPushButton(QIcon(":/help"), "Help");

	// type
	mapPlotType = new PVWidgets::PVAxisTypeWidget(this);
	// FIXME : We should populate *ModeWidget here.
	comboMapping = new PVWidgets::PVMappingModeWidget(this);
	comboPlotting = new PVWidgets::PVPlottingModeWidget(this);

	// tab parameter
	listTags = new PVXmlParamList("tag");
	btnTagHelp = new QPushButton(QIcon(":/help"), "Help");
	buttonColor = new PVXmlParamColorDialog("color", PVFORMAT_AXIS_COLOR_DEFAULT, this);
	colorLabel = new QLabel(tr("Color of the axis line :"));
	buttonTitleColor =
	    new PVXmlParamColorDialog("titlecolor", PVFORMAT_AXIS_TITLECOLOR_DEFAULT, this);
	titleColorLabel = new QLabel(tr("Color of the axis title :"));

	_layout_params_mp = new QHBoxLayout();
	_params_mapping = new PVWidgets::PVArgumentListWidget(
	    PVWidgets::PVArgumentListWidgetFactory::create_mapping_plotting_widget_factory(), this);
	_params_plotting = new PVWidgets::PVArgumentListWidget(
	    PVWidgets::PVArgumentListWidgetFactory::create_mapping_plotting_widget_factory(), this);

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

	// button next
	buttonNextAxis = new QPushButton(tr("Next"));
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::createTab
 *
 *****************************************************************************/
QVBoxLayout* PVInspector::PVXmlParamWidgetBoardAxis::createTab(const QString& title,
                                                               QTabWidget* tab)
{
	QWidget* tabWidget = new QWidget(tab);
	// create the layout
	QVBoxLayout* tabWidgetLayout = new QVBoxLayout(tabWidget);

	// creation of the tab
	tabWidgetLayout->setContentsMargins(0, 0, 0, 0);
	tabWidget->setLayout(tabWidgetLayout);

	// add the tab
	tab->addTab(tabWidget, title);

	// return the layout to add items
	return tabWidgetLayout;
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::draw
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardAxis::draw()
{

	// alloc
	QVBoxLayout* layoutParam = new QVBoxLayout();
	QVBoxLayout* tabGeneral = createTab("General", tabParam);
	QVBoxLayout* tabParameter = createTab("Parameter", tabParam);
	QWidget* widgetTabAndNext = new QWidget(this);
	QHBoxLayout* layoutRoot = new QHBoxLayout(this);

	// general layout
	setLayout(layoutRoot);
	layoutRoot->setContentsMargins(0, 0, 0, 0);
	// tab widget
	layoutRoot->addWidget(widgetTabAndNext);
	layoutParam->setContentsMargins(0, 0, 0, 0);
	widgetTabAndNext->setLayout(layoutParam);

	layoutParam->addWidget(tabParam);

	//***** tab general *****
	QGridLayout* gridLayout = new QGridLayout();
	int i = 0;
	// name
	gridLayout->addWidget(new QLabel(tr("Axis name :")), i, 0);
	gridLayout->addWidget(textName, i, 2, 1, -1);
	i += 2;
	// tag
	gridLayout->addWidget(new QLabel(tr("Tag :")), i, 0);
	gridLayout->addWidget(listTags, i, 2);
	gridLayout->addWidget(btnTagHelp, i, 4);
	i += 2;
	// type
	gridLayout->addWidget(new QLabel(tr("Type :")), i, 0);
	gridLayout->addWidget(mapPlotType, i, 2, 1, -1);
	i += 2;
	// type format
	gridLayout->addWidget(new QLabel(tr("Type Format:")), i, 0);
	gridLayout->addWidget(_type_format, i, 2);
	gridLayout->addWidget(btnTypeFormatHelp, i, 4);
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

	tabGeneral->addSpacerItem(
	    new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Expanding));

	//***** tab parameter *****
	gridLayout = new QGridLayout();
	i = 0;
	gridLayout->addWidget(colorLabel, i, 0);
	gridLayout->addWidget(buttonColor, i, 1);
	i += 2;
	gridLayout->addWidget(titleColorLabel, i, 0);
	gridLayout->addWidget(buttonTitleColor, i, 1);
	tabParameter->addLayout(gridLayout);
	tabParameter->addSpacerItem(
	    new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Expanding));

	// button next
	layoutParam->addWidget(buttonNextAxis);
	buttonNextAxis->setShortcut(QKeySequence(Qt::Key_Return));
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::initConnexion
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardAxis::initConnexion()
{

	connect(mapPlotType,
	        static_cast<void (QComboBox::*)(QString const&)>(&QComboBox::currentIndexChanged),
	        [this](QString const& text) {
		        updatePlotMapping(text);
		        node->setAttribute(QString(PVFORMAT_AXIS_TYPE_STR), mapPlotType->get_sel_type());
		        Q_EMIT signalRefreshView();
		    });

	connect(textName, &QLineEdit::textChanged, [this](QString const& text) {
		node->setAttribute(QString(PVFORMAT_AXIS_NAME_STR), text);
		Q_EMIT signalRefreshView();
	});

	connect(_type_format, &QLineEdit::textChanged, [this](QString const& text) {
		node->setAttribute(QString(PVFORMAT_AXIS_TYPE_FORMAT_STR), text);
		Q_EMIT signalRefreshView();
	});

	connect(comboMapping->get_combo_box(),
	        static_cast<void (QComboBox::*)(QString const&)>(&QComboBox::currentIndexChanged), this,
	        &PVInspector::PVXmlParamWidgetBoardAxis::updateMappingParams);
	connect(_params_mapping, SIGNAL(args_changed_Signal()), this, SLOT(slotSetParamsMapping()));

	connect(comboPlotting->get_combo_box(),
	        static_cast<void (QComboBox::*)(QString const&)>(&QComboBox::currentIndexChanged), this,
	        &PVInspector::PVXmlParamWidgetBoardAxis::updatePlottingParams);
	connect(_params_plotting, SIGNAL(args_changed_Signal()), this, SLOT(slotSetParamsPlotting()));

	connect(listTags, &QListWidget::itemSelectionChanged, [this]() {
		node->setAttribute(QString(PVFORMAT_AXIS_TAG_STR),
		                   listTags->selectedList().join(QString(QChar(PVFORMAT_TAGS_SEP))));
		Q_EMIT signalRefreshView();
	});

	connect(btnTagHelp, SIGNAL(clicked()), this, SLOT(slotShowTagHelp()));
	connect(btnTypeFormatHelp, SIGNAL(clicked()), this, SLOT(slotShowTypeFormatHelp()));

	// extra
	connect(buttonColor, &PVXmlParamColorDialog::changed, [this]() {
		node->setAttribute(QString(PVFORMAT_AXIS_COLOR_STR), buttonColor->getColor());
		Q_EMIT signalRefreshView();
	});

	connect(buttonTitleColor, &PVXmlParamColorDialog::changed, [this]() {
		node->setAttribute(QString(PVFORMAT_AXIS_TITLECOLOR_STR), buttonTitleColor->getColor());
		Q_EMIT signalRefreshView();
	});

	// button next axis
	connect(buttonNextAxis, SIGNAL(clicked()), this, SLOT(slotGoNextAxis()));
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::initValue
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardAxis::initValue()
{
	// init of combos

	// type ...  auto select and default value
	QString node_type = node->attribute(PVFORMAT_AXIS_TYPE_STR);
	if (node_type.isEmpty()) {
		node_type = PVFORMAT_AXIS_TYPE_DEFAULT;
	}
	mapPlotType->sel_type(node_type);

	// Select value from Xml. If Xml is invalid, it will keep default arguments
	_args_mapping.clear();
	Inendi::PVMappingFilter::p_type map_lib = get_mapping_lib_filter();
	QString node_mapping = node->getMappingProperties(map_lib->get_default_args(), _args_mapping);
	if (node_mapping.isEmpty()) {
		node_mapping = PVFORMAT_AXIS_MAPPING_DEFAULT;
	}
	comboMapping->set_mode(node_mapping);
	_args_map_mode[*map_lib] = _args_mapping;
	_params_mapping->set_args(_args_mapping);

	// Select value from Xml. If Xml is invalid, it will keep default arguments
	_args_plotting.clear();
	Inendi::PVPlottingFilter::p_type plot_lib = get_plotting_lib_filter();
	QString node_plotting =
	    node->getPlottingProperties(plot_lib->get_default_args(), _args_plotting);
	if (node_plotting.isEmpty()) {
		node_plotting = PVFORMAT_AXIS_PLOTTING_DEFAULT;
	}
	comboPlotting->set_mode(node_plotting);
	_args_plot_mode[*plot_lib] = _args_plotting;
	_params_plotting->set_args(_args_plotting);

	// extra
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

	setListTags();
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardAxis::getWidgetToFocus
 *
 *****************************************************************************/
QWidget* PVInspector::PVXmlParamWidgetBoardAxis::getWidgetToFocus()
{
	return (QWidget*)textName;
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::slotGoNextAxis
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardAxis::slotGoNextAxis()
{
	if (!node->isOnRoot) { // if we are not on root...
		Q_EMIT signalSelectNext();
	}
}

void PVInspector::PVXmlParamWidgetBoardAxis::slotSetParamsMapping()
{
	QString mode = comboMapping->get_mode();
	Inendi::PVMappingFilter::p_type lib_filter = get_mapping_lib_filter();
	_args_map_mode[*lib_filter] = _args_mapping;
	node->setMappingProperties(mode, lib_filter->get_default_args(), _args_mapping);
}

void PVInspector::PVXmlParamWidgetBoardAxis::slotSetParamsPlotting()
{
	QString mode = comboPlotting->get_mode();
	Inendi::PVPlottingFilter::p_type lib_filter = get_plotting_lib_filter();
	_args_plot_mode[*lib_filter] = _args_plotting;
	node->setPlottingProperties(mode, lib_filter->get_default_args(), _args_plotting);
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::updatePlotMapping
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardAxis::updatePlotMapping(const QString& t)
{
	if (t.length() > 1) {
		// Reset mapping/plotting (use on type change)
		QString type = mapPlotType->get_sel_type();
		comboMapping->clear();
		comboMapping->populate_from_type(type);
		comboMapping->select_default();
	}
}

Inendi::PVMappingFilter::p_type PVInspector::PVXmlParamWidgetBoardAxis::get_mapping_lib_filter()
{
	QString mode = comboMapping->get_mode();
	if (mode.isNull()) {
		// We update mapping on mapping widget change but when we change type,
		// mappingBox is cleared before we set new mapping possibilities and
		// mode becore empty.
		mode = PVFORMAT_AXIS_MAPPING_DEFAULT;
	}
	return LIB_CLASS(Inendi::PVMappingFilter)::get().get_class_by_name(mode);
}

Inendi::PVPlottingFilter::p_type PVInspector::PVXmlParamWidgetBoardAxis::get_plotting_lib_filter()
{
	QString mode = comboPlotting->get_mode();
	if (mode.isNull()) {
		// We update plotting on plotting widget change but when we change mapping,
		// plottingBox is cleared before we set new mapping possibilities and
		// mode becore empty.
		mode = PVFORMAT_AXIS_PLOTTING_DEFAULT;
	}
	return LIB_CLASS(Inendi::PVPlottingFilter)::get().get_class_by_name(mode);
}

void PVInspector::PVXmlParamWidgetBoardAxis::updateMappingParams()
{
	_args_mapping.clear();
	Inendi::PVMappingFilter::p_type lib_filter = get_mapping_lib_filter();
	auto it = _args_map_mode.find(*lib_filter);
	if (it != _args_map_mode.end()) {
		_args_mapping = it->second;
	} else {
		_args_mapping = lib_filter->get_default_args();
	}
	_params_mapping->set_args(_args_mapping);

	slotSetParamsMapping();

	comboPlotting->clear();
	comboPlotting->populate_from_type(mapPlotType->get_sel_type(), comboMapping->get_mode());
	comboPlotting->select_default();
}

void PVInspector::PVXmlParamWidgetBoardAxis::updatePlottingParams()
{
	_args_plotting.clear();
	Inendi::PVPlottingFilter::p_type lib_filter = get_plotting_lib_filter();
	auto it = _args_plot_mode.find(*lib_filter);
	if (it != _args_plot_mode.end()) {
		_args_plotting = it->second;
	} else {
		_args_plotting = lib_filter->get_default_args();
	}
	_params_plotting->set_args(_args_plotting);

	slotSetParamsPlotting();
}

void PVInspector::PVXmlParamWidgetBoardAxis::setListTags()
{
	listTags->clear();
	listTags->addItem(PVFORMAT_AXIS_TAG_DEFAULT);

	QSet<QString> list_tags = getListTags();
	QSet<QString> list_splitter_tags = getListParentSplitterTag();

	listTags->setItems(list_tags.unite(list_splitter_tags).toList());

	listTags->sortItems();

	QString node_tag = node->attribute(PVFORMAT_AXIS_TAG_STR);
	if (node_tag.isEmpty()) {
		node_tag = PVFORMAT_AXIS_TAG_DEFAULT;
	}
	listTags->select(node_tag.split(PVFORMAT_TAGS_SEP));
}

QSet<QString> PVInspector::PVXmlParamWidgetBoardAxis::getListTags()
{
	QSet<QString> ret;
	Inendi::PVLayerFilterListTags const& lt = LIB_CLASS(Inendi::PVLayerFilter)::get().get_tags();
	for (Inendi::PVLayerFilterTag const& tag : lt) {
		ret << (QString)tag;
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
	PVFilter::PVFieldsSplitter_p filter_p =
	    LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_class_by_name(
	        parent->attribute("type", ""));

	// Ok, get the tags !
	PVFilter::PVFieldsSplitterListTags const& tags =
	    LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_tags_for_class(*filter_p);
	for (int i = 0; i < tags.size(); i++) {
		PVFilter::PVFieldsSplitterTag const& tag = tags.at(i);
		ret << (QString)tag;
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

void PVInspector::PVXmlParamWidgetBoardAxis::slotShowTypeFormatHelp()
{
	PVWidgets::PVTimeFormatHelpDlg* dlg =
	    new PVWidgets::PVTimeFormatHelpDlg(_type_format, parentWidget());
	dlg->exec();
}
