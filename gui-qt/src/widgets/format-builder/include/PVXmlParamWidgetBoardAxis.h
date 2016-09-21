/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVXMLPARAMWIDGETBOARDAXIS_H
#define PVXMLPARAMWIDGETBOARDAXIS_H
#include <QWidget>
#include <QDir>
#include <QStringList>
#include <QRegExp>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QVariant>
#include <QDebug>
#include <QTableWidget>
#include <QTextEdit>
#include <QDateTime>
#include <QPushButton>
#include <QGroupBox>
#include <QTabWidget>
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
#include <inendi/widgets/PVAxisTypeWidget.h>
#include <inendi/widgets/PVMappingModeWidget.h>
#include <inendi/widgets/PVPlottingModeWidget.h>

#include <inendi/plugins.h>
#include <inendi/PVLayerFilter.h>
#include <inendi/PVMappingFilter.h>
#include <inendi/PVPlottingFilter.h>

namespace PVWidgets
{
class PVArgumentListWidget;
} // namespace PVWidgets

namespace PVInspector
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
	QVBoxLayout* createTab(const QString& title, QTabWidget* tab);
	void draw();
	void initConnexion();
	void initValue();
	void setListTags();
	Inendi::PVMappingFilter::p_type get_mapping_lib_filter();
	Inendi::PVPlottingFilter::p_type get_plotting_lib_filter();

	QSet<QString> getListTags();
	QSet<QString> getListParentSplitterTag();

	QStringList get_current_tags();
	/***************************  board items **********************/
	//***** tab general *****
	QTabWidget* tabParam;
	PVXmlParamWidgetEditorBox* textName; // name
	// type
	PVXmlParamWidgetEditorBox* _type_format; //!< Format to parse data (use for time)
	QPushButton* btnTypeFormatHelp;

	PVWidgets::PVAxisTypeWidget* mapPlotType;
	PVWidgets::PVMappingModeWidget* comboMapping;
	PVWidgets::PVPlottingModeWidget* comboPlotting;
	PVXmlParamList* listTags;

	//***** tab time format *****
	QLabel* timeFormatLabel;
	QCheckBox* useParentRegExpValue;
	QPushButton* btnTagHelp;

	//***** tab param *****
	PVXmlParamColorDialog* buttonColor;
	QLabel* colorLabel;
	PVXmlParamColorDialog* buttonTitleColor;
	QLabel* titleColorLabel;

	//***** view values from parent regexp *****
	QTextEdit* tableValueFromParentRegExp;

	// Mapping/plotting parameters widgets
	QHBoxLayout* _layout_params_mp;
	std::map<Inendi::PVMappingFilter::base_registrable, PVCore::PVArgumentList> _args_map_mode;
	std::map<Inendi::PVPlottingFilter::base_registrable, PVCore::PVArgumentList> _args_plot_mode;
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
	void slotShowTagHelp();
	void slotShowTypeFormatHelp();
	void updateMappingParams();
	void updatePlottingParams();
	void slotSetParamsMapping();
	void slotSetParamsPlotting();

  Q_SIGNALS:
	void signalRefreshView();
	void signalSelectNext();
};
} // namespace PVInspector
#endif /* PVXMLPARAMWIDGETBOARDAXIS_H */
