/**
 * \file PVXmlParamWidgetBoardAxis.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVXMLPARAMWIDGETBOARDAXIS_H
#define	PVXMLPARAMWIDGETBOARDAXIS_H
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
#include <picviz/widgets/PVAxisTypeWidget.h>
#include <picviz/widgets/PVMappingModeWidget.h>
#include <picviz/widgets/PVPlottingModeWidget.h>

#include <picviz/plugins.h>
#include <picviz/PVLayerFilter.h>
#include <picviz/PVMappingFilter.h>
#include <picviz/PVPlottingFilter.h>

namespace PVWidgets {
class PVArgumentListWidget;
}

namespace PVInspector{

class PVXmlParamWidget;

class PVXmlParamWidgetBoardAxis : public QWidget {
    Q_OBJECT
public:
    PVXmlParamWidgetBoardAxis(PVRush::PVXmlTreeNodeDom *pNode, PVXmlParamWidget* parent);
    virtual ~PVXmlParamWidgetBoardAxis();
    QWidget *getWidgetToFocus();
	PVXmlParamWidget* parent() { return _parent; }
    
  private:
    void allocBoardFields();
    QVBoxLayout *createTab(const QString &title, QTabWidget *tab);
    void disableConnexion();
    void disAllocBoardFields();
    void draw();
    void initConnexion();
    void initValue();
	void checkMappingTimeFormat();
	void setComboGroup();
	void setListTags();
	Picviz::PVMappingFilter::p_type get_mapping_lib_filter();
	Picviz::PVPlottingFilter::p_type get_plotting_lib_filter();
    
    QStringList listType() const;
    QStringList getListTypeMapping(const QString& mType);
    QStringList getListTypePlotting(const QString& mType);
	QSet<QString> getListTags();
	QSet<QString> getListParentSplitterTag();
    
	QStringList get_current_tags();
    /***************************  board items **********************/
    //***** tab general ***** 
    QTabWidget *tabParam;
    PVXmlParamWidgetEditorBox *textName;//name
    //type
	PVWidgets::PVAxisTypeWidget* mapPlotType;
	PVWidgets::PVMappingModeWidget* comboMapping;
	PVWidgets::PVPlottingModeWidget* comboPlotting;
	PVXmlParamComboBox * comboGroup;
	PVXmlParamList* listTags;
    
    //***** tab time format ***** 
    QLabel *timeFormatLabel;
    QCheckBox *useParentRegExpValue;
	QPushButton* btnGroupAdd;
	QPushButton* btnTagHelp;
    
    //***** tab param ***** 
    PVXmlParamWidgetEditorBox *group;
    QLabel *groupLabel;
    PVXmlParamColorDialog *buttonColor;
    QLabel *colorLabel;
    PVXmlParamColorDialog *buttonTitleColor;
    QLabel *titleColorLabel;
    
    //***** view values from parent regexp *****
    QTextEdit *tableValueFromParentRegExp;

	// Mapping/plotting parameters widgets
	QHBoxLayout* _layout_params_mp;
	std::map<Picviz::PVMappingFilter::base_registrable, PVCore::PVArgumentList> _args_map_mode;
	std::map<Picviz::PVPlottingFilter::base_registrable, PVCore::PVArgumentList> _args_plot_mode;
	PVCore::PVArgumentList _args_mapping;
	PVCore::PVArgumentList _args_plotting;
	PVWidgets::PVArgumentListWidget* _params_mapping;
	PVWidgets::PVArgumentListWidget* _params_plotting;
	QGroupBox* _grp_mapping;
	QGroupBox* _grp_plotting;

    
    QPushButton *buttonNextAxis;
    /***************************  board items **********************/
    
    
    //editing node
    PVRush::PVXmlTreeNodeDom *node;
    QString pluginListURL;

	PVXmlParamWidget* _parent;
    
public slots:
    void slotGoNextAxis();
    void slotSetValues();
    void updatePlotMapping(const QString& t) ;
	void slotAddGroup();
	void slotShowTagHelp();
	void updateMappingParams();
	void updatePlottingParams();
	void slotSetParamsMapping();
	void slotSetParamsPlotting();
    
    signals:
    void signalRefreshView();
    void signalSelectNext();
};
}
#endif	/* PVXMLPARAMWIDGETBOARDAXIS_H */

