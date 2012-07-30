/**
 * \file PVAxesCombinationDialog.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVAXESCOMBINATIONDIALOG_H
#define PVAXESCOMBINATIONDIALOG_H

#include <pvkernel/core/general.h>
#include <picviz/PVView_types.h>
#include <QDialog>

namespace PVInspector {

class PVMainWindow;
class PVTabSplitter;
class PVAxesCombinationWidget;

class PVAxesCombinationDialog: public QDialog
{
	Q_OBJECT

public:
	PVAxesCombinationDialog(Picviz::PVView_sp view, PVTabSplitter* tab, PVMainWindow* mw);

public:
	void save_current_combination();
	void update_used_axes();

protected slots:
	void refresh_axes_slot();
	void axes_count_changed_slot();
	void cancel_slot();

protected:
	PVMainWindow* main_window;
	PVTabSplitter* tab;
	PVAxesCombinationWidget* _axes_widget;
	Picviz::PVView_sp _view;
};

}

#endif
