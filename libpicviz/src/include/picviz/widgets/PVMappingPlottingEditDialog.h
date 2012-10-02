/**
 * \file PVMappingPlottingEditDialog.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVMAPPINGPLOTTINGEDITDIALOG_H
#define PVMAPPINGPLOTTINGEDITDIALOG_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVFormat_types.h>
#include <picviz/PVPtrObjects.h>
#include <picviz/PVAxesCombination.h>

#include <QDialog>
#include <QComboBox>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QVBoxLayout>

namespace Picviz {
class PVMapping;
class PVPlotting;
}

namespace PVWidgets {

class PVMappingPlottingEditDialog: public QDialog
{
	Q_OBJECT
public:
	PVMappingPlottingEditDialog(Picviz::PVMapping* mapping, Picviz::PVPlotting* plotting, QWidget* parent = NULL);
	virtual ~PVMappingPlottingEditDialog();

private:
	inline bool has_mapping() const { return _mapping != NULL; };
	inline bool has_plotting() const { return _plotting != NULL; }

	void init_layout();
	void finish_layout();
	void load_settings();
	void reset_settings_with_format();

	static QLabel* create_label(QString const& text, Qt::Alignment align = Qt::AlignCenter);

private slots:
	void type_changed(const QString& type);
	void save_settings();

private:
	QGridLayout* _main_grid;
	QVBoxLayout* _main_layout;
	QLineEdit* _edit_name;
	Picviz::PVMapping* _mapping;
	Picviz::PVPlotting* _plotting;
	const Picviz::PVAxesCombination::list_axes_t* _axes;
};

}

#endif
