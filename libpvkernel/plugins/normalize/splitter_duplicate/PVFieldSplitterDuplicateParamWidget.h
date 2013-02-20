/**
 * \file PVFieldSplitterDuplicateParamWidget.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef PVFIELDSPLITTERDUPLICATEPARAMWIDGET_H
#define PVFIELDSPLITTERDUPLICATEPARAMWIDGET_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVFieldsFilterParamWidget.h>
#include <boost/shared_ptr.hpp>

#include <QSpinBox>
#include <QWidget>
#include <QAction>

#include <pvkernel/widgets/qkeysequencewidget.h>

namespace PVFilter {

class PVFieldSplitterDuplicateParamWidget: public PVFieldsSplitterParamWidget
{
	Q_OBJECT

public:
	PVFieldSplitterDuplicateParamWidget();

public:
	QAction* get_action_menu();
	QWidget* get_param_widget();

	size_t force_number_children() {
		return 0;
	}

private slots:
	void updateNChilds(int n);

private:
	QAction* _action_menu;
	QWidget* _param_widget;
	QSpinBox* _duplications_spin_box;

private:
	CLASS_REGISTRABLE_NOCOPY(PVFieldSplitterDuplicateParamWidget)
};

}

#endif // PVFIELDSPLITTERDUPLICATEPARAMWIDGET_H
