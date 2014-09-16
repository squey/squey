/**
 * \file PVFieldSplitterDnsFqdnParamWidget.h
 *
 * Copyright (C) Picviz Labs 2014
 */

#ifndef PVFIELDSPLITTERDNSFQDNPARAMWIDGET_H
#define PVFIELDSPLITTERDNSFQDNPARAMWIDGET_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVFieldsFilterParamWidget.h>

#include <QAction>
#include <QWidget>
#include <QCheckBox>

#include <pvkernel/widgets/qkeysequencewidget.h>

namespace PVFilter {

class PVFieldSplitterDnsFqdnParamWidget: public PVFieldsSplitterParamWidget
{
	Q_OBJECT

public:
	PVFieldSplitterDnsFqdnParamWidget();

	size_t force_number_children()
	{
		return 0;
	}
public:
	QAction* get_action_menu();
	QWidget* get_param_widget();

private:
	void update_args(PVCore::PVArgumentList& args);

private slots:
	void split_cb_changed(int state);
	void rev_cb_changed(int state);

private:
	QAction*   _action_menu;
	QWidget*   _param_widget;
	QCheckBox* _split_cb[6];
	QCheckBox* _rev_cb[3];
	int        _n;

private:
	CLASS_REGISTRABLE_NOCOPY(PVFieldSplitterDnsFqdnParamWidget)
};

}

#endif // PVFIELDSPLITTERDNSFQDNPARAMWIDGET_H
