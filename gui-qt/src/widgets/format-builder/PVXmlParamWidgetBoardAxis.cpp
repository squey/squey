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
#include <pvkernel/widgets/PVArgumentListWidget.h>
#include <inendi/widgets/PVArgumentListWidgetFactory.h>
#include <pvkernel/widgets/editors/PVTimeFormatEditor.h>

#include <QDialogButtonBox>
#include <QFormLayout>

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
	initValue();
	initConnexion();
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::allocBoardFields
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardAxis::allocBoardFields()
{
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
	buttonColor = new PVXmlParamColorDialog("color", PVFORMAT_AXIS_COLOR_DEFAULT, this);
	buttonTitleColor =
	    new PVXmlParamColorDialog("titlecolor", PVFORMAT_AXIS_TITLECOLOR_DEFAULT, this);

	_layout_params_mp = new QHBoxLayout();
	_params_mapping = new PVWidgets::PVArgumentListWidget(
	    PVWidgets::PVArgumentListWidgetFactory::create_mapping_plotting_widget_factory(), this);
	_params_plotting = new PVWidgets::PVArgumentListWidget(
	    PVWidgets::PVArgumentListWidgetFactory::create_mapping_plotting_widget_factory(), this);

	QSizePolicy grp_params_policy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	grp_params_policy.setHorizontalStretch(1);
	_grp_mapping = new QGroupBox(tr("Mapping properties"));
	_grp_mapping->setVisible(false);
	_grp_plotting = new QGroupBox(tr("Plotting properties"));
	_grp_plotting->setVisible(false);
	_params_mapping->setSizePolicy(grp_params_policy);
	_params_mapping->setSizePolicy(grp_params_policy);
	_params_plotting->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	auto tmp_layout = new QVBoxLayout();
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
 * PVInspector::PVXmlParamWidgetBoardAxis::draw
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardAxis::draw()
{
	auto layoutRoot = new QVBoxLayout(this);

	setLayout(layoutRoot);

	auto general = new QWidget(this);
	auto form_layout = new QFormLayout();

	form_layout->addRow(tr("Axis name :"), textName);
	form_layout->addRow(tr("Type :"), mapPlotType);

	auto tf_widget = new QWidget();
	auto tf_layout = new QHBoxLayout();
	tf_layout->setContentsMargins(0, 0, 0, 0);
	tf_layout->addWidget(_type_format);
	tf_layout->addWidget(btnTypeFormatHelp);
	tf_widget->setLayout(tf_layout);

	form_layout->addRow(tr("Type Format:"), tf_widget);
	form_layout->addRow(tr("Mapping :"), comboMapping);
	form_layout->addRow(tr("Plotting :"), comboPlotting);

	form_layout->addRow(tr("Color of the axis line :"), buttonColor);
	form_layout->addRow(tr("Color of the axis title :"), buttonTitleColor);

	general->setLayout(form_layout);
	layoutRoot->addWidget(general);

	// Mapping/plotting properties
	auto prop_widget = new QWidget();
	_layout_params_mp->addWidget(_grp_mapping);
	_layout_params_mp->addWidget(_grp_plotting);
	prop_widget->setLayout(_layout_params_mp);
	layoutRoot->addWidget(prop_widget);

	layoutRoot->addSpacerItem(
	    new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Expanding));

	// button next
	layoutRoot->addWidget(buttonNextAxis);
	buttonNextAxis->setShortcut(QKeySequence(Qt::Key_Return));
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardAxis::initConnexion
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardAxis::initConnexion()
{

	connect(mapPlotType, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged),
	        [this](int) {
		        updatePlotMapping();
		        node->setAttribute(QString(PVFORMAT_AXIS_TYPE_STR), mapPlotType->get_sel_type());
		        if (mapPlotType->get_sel_type() == "string") {
			        node->setAttribute(QString(PVFORMAT_AXIS_MAPPING_STR), "string");
			        comboMapping->set_mode("string");
		        }
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
	        static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this,
	        &PVInspector::PVXmlParamWidgetBoardAxis::updateMappingParams);
	connect(_params_mapping, &PVWidgets::PVArgumentListWidget::args_changed_Signal, this,
	        &PVInspector::PVXmlParamWidgetBoardAxis::slotSetParamsMapping);

	connect(comboPlotting->get_combo_box(),
	        static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this,
	        &PVInspector::PVXmlParamWidgetBoardAxis::updatePlottingParams);
	connect(_params_plotting, &PVWidgets::PVArgumentListWidget::args_changed_Signal, this,
	        &PVInspector::PVXmlParamWidgetBoardAxis::slotSetParamsPlotting);

	connect(btnTypeFormatHelp, &QAbstractButton::clicked, this,
	        &PVInspector::PVXmlParamWidgetBoardAxis::slotShowTypeFormatHelp);

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
	connect(buttonNextAxis, &QAbstractButton::clicked, this,
	        &PVInspector::PVXmlParamWidgetBoardAxis::slotGoNextAxis);
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
	updatePlotMapping();

	// Select value from Xml. If Xml is invalid, it will keep default arguments
	_args_mapping.clear();
	Inendi::PVMappingFilter::p_type map_lib = get_mapping_lib_filter();
	QString node_mapping = node->getMappingProperties(map_lib->get_default_args(), _args_mapping);
	if (node_mapping.isEmpty()) {
		if (node_type == "string") {
			comboMapping->set_mode("string");
		} else {
			comboMapping->select_default();
		}
	} else {
		comboMapping->set_mode(node_mapping);
	}

	_args_map_mode[*map_lib] = _args_mapping;
	_params_mapping->set_args(_args_mapping);

	updateMappingParams();

	// Select value from Xml. If Xml is invalid, it will keep default arguments
	_args_plotting.clear();
	Inendi::PVPlottingFilter::p_type plot_lib = get_plotting_lib_filter();
	QString node_plotting =
	    node->getPlottingProperties(plot_lib->get_default_args(), _args_plotting);
	if (node_plotting.isEmpty()) {
		comboPlotting->select_default();
	} else {
		comboPlotting->set_mode(node_plotting);
	}
	_args_plot_mode[*plot_lib] = _args_plotting;
	_params_plotting->set_args(_args_plotting);
	updatePlottingParams();

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
void PVInspector::PVXmlParamWidgetBoardAxis::updatePlotMapping()
{
	// Reset mapping/plotting (use on type change)
	QString type = mapPlotType->get_sel_type();
	comboMapping->clear();
	comboMapping->populate_from_type(type);
	comboMapping->select_default();
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

void PVInspector::PVXmlParamWidgetBoardAxis::slotShowTypeFormatHelp()
{
	auto dlg = new PVWidgets::PVTimeFormatHelpDlg(_type_format, parentWidget());
	dlg->exec();
}
