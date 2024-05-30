//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <squey/PVMappingFilter.h>
#include <squey/PVScalingFilter.h>

#include <PVXmlParamWidgetBoardAxis.h>
#include <PVFormatBuilderWidget.h>
#include <pvkernel/widgets/PVArgumentListWidget.h>
#include <squey/widgets/PVArgumentListWidgetFactory.h>
#include <pvkernel/widgets/editors/PVTimeFormatEditor.h>

#include <QDialogButtonBox>
#include <QFormLayout>

/******************************************************************************
 *
 * App::PVXmlParamWidgetBoardAxis::PVXmlParamWidgetBoardAxis
 *
 *****************************************************************************/
App::PVXmlParamWidgetBoardAxis::PVXmlParamWidgetBoardAxis(PVRush::PVXmlTreeNodeDom* pNode,
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
 * App::PVXmlParamWidgetBoardAxis::allocBoardFields
 *
 *****************************************************************************/
void App::PVXmlParamWidgetBoardAxis::allocBoardFields()
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
	comboScaling = new PVWidgets::PVScalingModeWidget(this);

	// tab parameter
	buttonColor = new PVXmlParamColorDialog("color", PVFORMAT_AXIS_COLOR_DEFAULT, this);
	buttonTitleColor =
	    new PVXmlParamColorDialog("titlecolor", PVFORMAT_AXIS_TITLECOLOR_DEFAULT, this);

	_layout_params_mp = new QHBoxLayout();
	_params_mapping = new PVWidgets::PVArgumentListWidget(
	    PVWidgets::PVArgumentListWidgetFactory::create_mapping_scaling_widget_factory(), this);
	_params_scaling = new PVWidgets::PVArgumentListWidget(
	    PVWidgets::PVArgumentListWidgetFactory::create_mapping_scaling_widget_factory(), this);

	QSizePolicy grp_params_policy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	grp_params_policy.setHorizontalStretch(1);
	_grp_mapping = new QGroupBox(tr("Mapping properties"));
	_grp_mapping->setVisible(false);
	_grp_scaling = new QGroupBox(tr("Scaling properties"));
	_grp_scaling->setVisible(false);
	_params_mapping->setSizePolicy(grp_params_policy);
	_params_mapping->setSizePolicy(grp_params_policy);
	_params_scaling->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	auto tmp_layout = new QVBoxLayout();
	tmp_layout->addWidget(_params_mapping);
	_grp_mapping->setLayout(tmp_layout);
	tmp_layout = new QVBoxLayout();
	tmp_layout->addWidget(_params_scaling);
	_grp_scaling->setLayout(tmp_layout);

	// button next
	buttonNextAxis = new QPushButton(tr("Next"));
}

/******************************************************************************
 *
 * App::PVXmlParamWidgetBoardAxis::draw
 *
 *****************************************************************************/
void App::PVXmlParamWidgetBoardAxis::draw()
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
	form_layout->addRow(tr("Scaling :"), comboScaling);

	form_layout->addRow(tr("Color of the axis line :"), buttonColor);
	form_layout->addRow(tr("Color of the axis title :"), buttonTitleColor);

	general->setLayout(form_layout);
	layoutRoot->addWidget(general);

	// Mapping/scaling properties
	auto prop_widget = new QWidget();
	_layout_params_mp->addWidget(_grp_mapping);
	_layout_params_mp->addWidget(_grp_scaling);
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
 * App::PVXmlParamWidgetBoardAxis::initConnexion
 *
 *****************************************************************************/
void App::PVXmlParamWidgetBoardAxis::initConnexion()
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
	        &App::PVXmlParamWidgetBoardAxis::updateMappingParams);
	connect(_params_mapping, &PVWidgets::PVArgumentListWidget::args_changed_Signal, this,
	        &App::PVXmlParamWidgetBoardAxis::slotSetParamsMapping);

	connect(comboScaling->get_combo_box(),
	        static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this,
	        &App::PVXmlParamWidgetBoardAxis::updateScalingParams);
	connect(_params_scaling, &PVWidgets::PVArgumentListWidget::args_changed_Signal, this,
	        &App::PVXmlParamWidgetBoardAxis::slotSetParamsScaling);

	connect(btnTypeFormatHelp, &QAbstractButton::clicked, this,
	        &App::PVXmlParamWidgetBoardAxis::slotShowTypeFormatHelp);

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
	        &App::PVXmlParamWidgetBoardAxis::slotGoNextAxis);
}

/******************************************************************************
 *
 * App::PVXmlParamWidgetBoardAxis::initValue
 *
 *****************************************************************************/
void App::PVXmlParamWidgetBoardAxis::initValue()
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
	Squey::PVMappingFilter::p_type map_lib = get_mapping_lib_filter();
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
	_args_scaling.clear();
	Squey::PVScalingFilter::p_type plot_lib = get_scaling_lib_filter();
	QString node_scaling =
	    node->getScalingProperties(plot_lib->get_default_args(), _args_scaling);
	if (node_scaling.isEmpty()) {
		comboScaling->select_default();
	} else {
		comboScaling->set_mode(node_scaling);
	}
	_args_plot_mode[*plot_lib] = _args_scaling;
	_params_scaling->set_args(_args_scaling);
	updateScalingParams();

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
 * void App::PVXmlParamWidgetBoardAxis::getWidgetToFocus
 *
 *****************************************************************************/
QWidget* App::PVXmlParamWidgetBoardAxis::getWidgetToFocus()
{
	return (QWidget*)textName;
}

/******************************************************************************
 *
 * App::PVXmlParamWidgetBoardAxis::slotGoNextAxis
 *
 *****************************************************************************/
void App::PVXmlParamWidgetBoardAxis::slotGoNextAxis()
{
	if (!node->isOnRoot) { // if we are not on root...
		Q_EMIT signalSelectNext();
	}
}

void App::PVXmlParamWidgetBoardAxis::slotSetParamsMapping()
{
	QString mode = comboMapping->get_mode();
	Squey::PVMappingFilter::p_type lib_filter = get_mapping_lib_filter();
	_args_map_mode[*lib_filter] = _args_mapping;
	node->setMappingProperties(mode, lib_filter->get_default_args(), _args_mapping);
}

void App::PVXmlParamWidgetBoardAxis::slotSetParamsScaling()
{
	QString mode = comboScaling->get_mode();
	Squey::PVScalingFilter::p_type lib_filter = get_scaling_lib_filter();
	_args_plot_mode[*lib_filter] = _args_scaling;
	node->setScalingProperties(mode, lib_filter->get_default_args(), _args_scaling);
}

/******************************************************************************
 *
 * App::PVXmlParamWidgetBoardAxis::updatePlotMapping
 *
 *****************************************************************************/
void App::PVXmlParamWidgetBoardAxis::updatePlotMapping()
{
	// Reset mapping/scaling (use on type change)
	QString type = mapPlotType->get_sel_type();
	comboMapping->clear();
	comboMapping->populate_from_type(type);
	comboMapping->select_default();
}

Squey::PVMappingFilter::p_type App::PVXmlParamWidgetBoardAxis::get_mapping_lib_filter()
{
	QString mode = comboMapping->get_mode();
	if (mode.isNull()) {
		// We update mapping on mapping widget change but when we change type,
		// mappingBox is cleared before we set new mapping possibilities and
		// mode becore empty.
		mode = PVFORMAT_AXIS_MAPPING_DEFAULT;
	}
	return LIB_CLASS(Squey::PVMappingFilter)::get().get_class_by_name(mode);
}

Squey::PVScalingFilter::p_type App::PVXmlParamWidgetBoardAxis::get_scaling_lib_filter()
{
	QString mode = comboScaling->get_mode();
	if (mode.isNull()) {
		// We update scaling on scaling widget change but when we change mapping,
		// scalingBox is cleared before we set new mapping possibilities and
		// mode becore empty.
		mode = PVFORMAT_AXIS_SCALING_DEFAULT;
	}
	return LIB_CLASS(Squey::PVScalingFilter)::get().get_class_by_name(mode);
}

void App::PVXmlParamWidgetBoardAxis::updateMappingParams()
{
	_args_mapping.clear();
	Squey::PVMappingFilter::p_type lib_filter = get_mapping_lib_filter();
	auto it = _args_map_mode.find(*lib_filter);
	if (it != _args_map_mode.end()) {
		_args_mapping = it->second;
	} else {
		_args_mapping = lib_filter->get_default_args();
	}
	_params_mapping->set_args(_args_mapping);

	slotSetParamsMapping();

	comboScaling->clear();
	comboScaling->populate_from_type(mapPlotType->get_sel_type(), comboMapping->get_mode());
	comboScaling->select_default();
}

void App::PVXmlParamWidgetBoardAxis::updateScalingParams()
{
	_args_scaling.clear();
	Squey::PVScalingFilter::p_type lib_filter = get_scaling_lib_filter();
	auto it = _args_plot_mode.find(*lib_filter);
	if (it != _args_plot_mode.end()) {
		_args_scaling = it->second;
	} else {
		_args_scaling = lib_filter->get_default_args();
	}
	_params_scaling->set_args(_args_scaling);

	slotSetParamsScaling();
}

void App::PVXmlParamWidgetBoardAxis::slotShowTypeFormatHelp()
{
	auto dlg = new PVWidgets::PVTimeFormatHelpDlg(_type_format, parentWidget());
	dlg->exec();
}
