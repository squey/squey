/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVXMLPARAMWIDGETBOARDAXIS_H
#define PVXMLPARAMWIDGETBOARDAXIS_H
#include <QWidget>
#include <QDir>
#include <QStringList>
#include <QRegExp>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QVariant>
#include <QDebug>
#include <QTextEdit>
#include <QDateTime>
#include <QPushButton>
#include <QGroupBox>
#include <QCheckBox>

#include <map>

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/rush/PVXmlTreeNodeDom.h>
#include <PVXmlParamWidgetEditorBox.h>
#include <PVXmlParamTextEdit.h>
#include <PVXmlParamComboBox.h>
#include <PVXmlParamColorDialog.h>
#include <PVXmlParamList.h>

// Widget helpers
#include <squey/widgets/PVAxisTypeWidget.h>
#include <squey/widgets/PVMappingModeWidget.h>
#include <squey/widgets/PVPlottingModeWidget.h>

#include <squey/plugins.h>
#include <squey/PVLayerFilter.h>
#include <squey/PVMappingFilter.h>
#include <squey/PVPlottingFilter.h>

namespace PVWidgets
{
class PVArgumentListWidget;
} // namespace PVWidgets

namespace App
{

class PVXmlParamWidget;

class PVXmlParamWidgetBoardAxis : public QWidget
{
	Q_OBJECT
  public:
	PVXmlParamWidgetBoardAxis(PVRush::PVXmlTreeNodeDom* pNode, PVXmlParamWidget* parent);
	QWidget* getWidgetToFocus();
	PVXmlParamWidget* parent() { return _parent; }

  private:
	void allocBoardFields();
	void draw();
	void initConnexion();
	void initValue();
	Squey::PVMappingFilter::p_type get_mapping_lib_filter();
	Squey::PVPlottingFilter::p_type get_plotting_lib_filter();

	/***************************  board items **********************/
	//***** tab general *****
	PVXmlParamWidgetEditorBox* textName; // name
	// type
	PVXmlParamWidgetEditorBox* _type_format; //!< Format to parse data (use for time)
	QPushButton* btnTypeFormatHelp;

	PVWidgets::PVAxisTypeWidget* mapPlotType;
	PVWidgets::PVMappingModeWidget* comboMapping;
	PVWidgets::PVPlottingModeWidget* comboPlotting;

	//***** tab time format *****
	QCheckBox* useParentRegExpValue;

	//***** tab param *****
	PVXmlParamColorDialog* buttonColor;
	PVXmlParamColorDialog* buttonTitleColor;

	//***** view values from parent regexp *****
	QTextEdit* tableValueFromParentRegExp;

	// Mapping/plotting parameters widgets
	QHBoxLayout* _layout_params_mp;
	std::map<Squey::PVMappingFilter::base_registrable, PVCore::PVArgumentList> _args_map_mode;
	std::map<Squey::PVPlottingFilter::base_registrable, PVCore::PVArgumentList> _args_plot_mode;
	PVCore::PVArgumentList _args_mapping;
	PVCore::PVArgumentList _args_plotting;
	PVWidgets::PVArgumentListWidget* _params_mapping;
	PVWidgets::PVArgumentListWidget* _params_plotting;
	QGroupBox* _grp_mapping;
	QGroupBox* _grp_plotting;

	QPushButton* buttonNextAxis;
	/***************************  board items **********************/

	// editing node
	PVRush::PVXmlTreeNodeDom* node;
	QString pluginListURL;

	PVXmlParamWidget* _parent;

  public Q_SLOTS:
	void slotGoNextAxis();
	void updatePlotMapping();
	void slotShowTypeFormatHelp();
	void updateMappingParams();
	void updatePlottingParams();
	void slotSetParamsMapping();
	void slotSetParamsPlotting();

  Q_SIGNALS:
	void signalRefreshView();
	void signalSelectNext();
};
} // namespace App
#endif /* PVXMLPARAMWIDGETBOARDAXIS_H */
