#ifndef PVMAPPINGPLOTTINGEDITDIALOG_H
#define PVMAPPINGPLOTTINGEDITDIALOG_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVFormat_types.h>
#include <picviz/PVPtrObjects.h>

#include <QDialog>
#include <QComboBox>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QVBoxLayout>

namespace PVInspector {

class PVMappingPlottingEditDialog: public QDialog
{
	Q_OBJECT
public:
	PVMappingPlottingEditDialog(Picviz::PVMapping* mapping, Picviz::PVPlotting* plotting, QWidget* parent = NULL);
	virtual ~PVMappingPlottingEditDialog();

private:
	inline bool has_mapping() const { return _mapping != NULL; };
	inline bool has_plotting() const { return _plotting != NULL; }

	static QStringList get_list_types();
	static QStringList get_list_mapping(QString const& type);
	static QStringList get_list_plotting(QString const& type);

	void init_layout();
	void finish_layout();
	void load_settings();
	void reset_settings_with_format();

	static QComboBox* init_combo(QStringList const& list, QString const& sel);
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
	const PVRush::PVFormat* _format;
};

}

#endif
