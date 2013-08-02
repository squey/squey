/**
 * \file PVFieldSplitterIPv4IPv6FromGUIDParamWidget.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef PVFIELDSPLITTERIPV4IPVFROMGUIDARAMWIDGET_H
#define PVFIELDSPLITTERIPV4IPVFROMGUIDARAMWIDGET_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVFieldsFilterParamWidget.h>
#include <boost/shared_ptr.hpp>

class QWidget;
class QAction;
class QCheckBox;
class QComboBox;

#include <pvkernel/widgets/qkeysequencewidget.h>

namespace PVFilter {

class PVFieldSplitterIPv4IPv6FromGUIDParamWidget: public PVFieldsSplitterParamWidget
{
	Q_OBJECT

public:
	PVFieldSplitterIPv4IPv6FromGUIDParamWidget();

public:
	QAction* get_action_menu();
	QWidget* get_param_widget();

	size_t force_number_children() {
		return 0;
	}

private slots:
	void updateNChilds();

private:
	QAction* _action_menu;
	QWidget* _param_widget;

	QCheckBox* _ipv4_cb;
	QCheckBox* _ipv6_cb;

private:
	CLASS_REGISTRABLE_NOCOPY(PVFieldSplitterIPv4IPv6FromGUIDParamWidget)
};

}

#endif // PVFIELDSPLITTERIPV4IPVFROMGUIDARAMWIDGET_H
