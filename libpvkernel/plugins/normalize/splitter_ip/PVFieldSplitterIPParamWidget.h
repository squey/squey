/**
 * \file PVFieldSplitterIPParamWidget.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef PVFIELDSPLITTERIPPARAMWIDGET_H
#define PVFIELDSPLITTERIPPARAMWIDGET_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVFieldsFilterParamWidget.h>
#include <boost/shared_ptr.hpp>

class QWidget;
class QAction;
class QCheckBox;
class QRadioButton;
class QComboBox;
class QLabel;
#include <QList>

#include <pvkernel/widgets/qkeysequencewidget.h>

namespace PVFilter {

class PVFieldSplitterIPParamWidget: public PVFieldsSplitterParamWidget
{
	Q_OBJECT

public:
	PVFieldSplitterIPParamWidget();

public:
	QAction* get_action_menu();
	QWidget* get_param_widget();

private slots:
	void set_ip_type();
	void update_child_count();

private:
	void set_groups_check_state();

private:
	QAction* _action_menu;

	QRadioButton* _ipv4;
	QRadioButton* _ipv6;

	QList<QCheckBox*> _cb_list;
	QList<QLabel*> _label_list;

	size_t _group_count;

private:
	CLASS_REGISTRABLE_NOCOPY(PVFieldSplitterIPParamWidget)
};

}

#endif // PVFIELDSPLITTERIPPARAMWIDGET_H
