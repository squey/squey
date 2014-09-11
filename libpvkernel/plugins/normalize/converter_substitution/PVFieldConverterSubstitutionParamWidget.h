/**
 * \file PVFieldConverterSubstitutionParamWidget.h
 *
 * Copyright (C) Picviz Labs 2014
 */

#ifndef PVFIELDSUBSTITUTIONPARAMWIDGET_H
#define PVFIELDSUBSTITUTIONPARAMWIDGET_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVFieldsFilterParamWidget.h>
#include <boost/shared_ptr.hpp>

class QAction;
class QCheckBox;
class QLineEdit;
class QWidget;

#include <pvkernel/widgets/qkeysequencewidget.h>

namespace PVWidgets {
	class QKeySequenceWidget;
}

namespace PVFilter {

class PVFieldConverterSubstitutionParamWidget: public PVFieldsConverterParamWidget
{
	Q_OBJECT

public:
	PVFieldConverterSubstitutionParamWidget();

public:
	QAction* get_action_menu();
	QWidget* get_param_widget();

private slots:
	void update_params();
	void browse_conversion_file();
	void use_default_value_checkbox_changed(int state);

private:
	QAction* _action_menu;
	QWidget* _param_widget;
	QLineEdit* _file_path_line_edit;
	QLineEdit* _default_value_line_edit;
	QCheckBox* _use_default_value_checkbox;
	PVWidgets::QKeySequenceWidget* _separator_char;
	PVWidgets::QKeySequenceWidget* _quote_char;

private:
	CLASS_REGISTRABLE_NOCOPY(PVFieldConverterSubstitutionParamWidget)
};

}

#endif // PVFIELDSUBSTITUTIONPARAMWIDGET_H
