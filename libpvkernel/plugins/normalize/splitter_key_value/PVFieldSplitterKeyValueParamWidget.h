/**
 * \file PVFieldSplitterKeyValueParamWidget.h
 *
 * Copyright (C) Picviz Labs 2014
 */

#ifndef PVFIELDSPLITTERKEYVALUEPARAMWIDGET_H
#define PVFIELDSPLITTERKEYVALUEPARAMWIDGET_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVFieldsFilterParamWidget.h>
#include <boost/shared_ptr.hpp>

class QAction;
class QWidget;
class QListWidget;
class QLineEdit;

#include <pvkernel/widgets/qkeysequencewidget.h>

namespace PVFilter {

class PVFieldSplitterKeyValueParamWidget: public PVFieldsSplitterParamWidget
{
	Q_OBJECT

public:
	PVFieldSplitterKeyValueParamWidget();

public:
	QAction* get_action_menu();
	QWidget* get_param_widget();

private slots:
	void update_params();
	void add_new_key();
	void del_keys();
	void move_key_down();
	void move_key_up();
	void update_children_count();

private:
	QAction* _action_menu;
	QWidget* _param_widget;

	PVWidgets::QKeySequenceWidget* _quote_char;
	QLineEdit* _separator_char_lineedit;
	QLineEdit* _affectation_operator_lineedit;

	QListWidget* _keys_list;

private:
	CLASS_REGISTRABLE_NOCOPY(PVFieldSplitterKeyValueParamWidget)
};

}

#endif // PVFIELDSPLITTERKEYVALUEPARAMWIDGET_H
