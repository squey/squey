/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVMAPPINGPLOTTINGEDITDIALOG_H
#define PVMAPPINGPLOTTINGEDITDIALOG_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVFormat_types.h>
#include <inendi/PVPtrObjects.h>
#include <inendi/PVAxesCombination.h>

#include <QDialog>
#include <QComboBox>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QVBoxLayout>

class QScrollArea;
class QGroupBox;

namespace Inendi {
class PVMapping;
class PVPlotting;
}

namespace PVWidgets {

class PVMappingPlottingEditDialog: public QDialog
{
	Q_OBJECT
public:
	PVMappingPlottingEditDialog(Inendi::PVMapping* mapping, Inendi::PVPlotting* plotting, QWidget* parent = NULL);
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
	QScrollArea* _main_scroll_area;
	QGroupBox* _main_group_box;
	Inendi::PVMapping* _mapping;
	Inendi::PVPlotting* _plotting;
	const Inendi::PVAxesCombination::list_axes_t* _axes;

};

}

#endif
