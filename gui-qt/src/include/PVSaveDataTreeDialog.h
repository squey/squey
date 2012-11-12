/**
 * \file PVSaveDataTreeDialog.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVSAVEDATATREEDIALOG_H
#define PVSAVEDATATREEDIALOG_H

#include <pvkernel/core/PVSerializeArchiveOptions_types.h>

#include <QCheckBox>
#include <QFileDialog>
#include <QWidget>

namespace PVInspector {

class PVSaveDataTreeDialog: public QFileDialog
{
	Q_OBJECT
public:
	PVSaveDataTreeDialog(PVCore::PVSerializeArchiveOptions_p options, QString const& suffix, QString const& filter, QWidget* parent);

protected slots:
	void include_files_Slot(int state);
	void tab_changed_Slot(int idx);

protected:
	PVCore::PVSerializeArchiveOptions& _options;
	QCheckBox* _save_everything_checkbox;
};

}

#endif
