/**
 * \file PVExpandSelDlg.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVEXPANDSELDLG_H
#define PVEXPANDSELDLG_H

#include <QDialog>
#include <QComboBox>
#include <QWidget>
#include <QDialogButtonBox>

#include <pvkernel/core/PVAxesIndexType.h>
#include <picviz/PVView_types.h>

#include <picviz/widgets/editors/PVAxesIndexEditor.h>

namespace PVInspector {

class PVExpandSelDlg: public QDialog
{
	Q_OBJECT
public:
	PVExpandSelDlg(Picviz::PVView_p view, QWidget* parent);

public:
	PVCore::PVAxesIndexType get_axes() const;
	QString get_mode();

private slots:
	void update_list_modes();

private:
	Picviz::PVView const& _view;
	PVWidgets::PVAxesIndexEditor* _axes_editor;
	QComboBox* _combo_modes;
	QDialogButtonBox* _btns;
};

}

#endif
