/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVIMPORTFILEDIALOG_H
#define PVIMPORTFILEDIALOG_H

#include <QComboBox>
#include <QCheckBox>
#include <QLineEdit>
#include <QFileDialog>

#include <inendi/plugins.h>

namespace PVInspector
{

/**
 * \class PVImportFileDialog
 */
class PVImportFileDialog : public QFileDialog
{
	Q_OBJECT

	QComboBox* treat_as_combobox;

  public:
	PVImportFileDialog(QWidget* parent = 0);

	QStringList normalized_plugins_list;

	QStringList getFileNames(QString& treat_as);
	void setDefaults();
	QCheckBox* activate_netflow_checkbox;
	QLineEdit* from_line_edit;
	QLineEdit* to_line_edit;
	// public slots:
	//    void new_file_Slot();
	//    void new_scene_Slot();
	//    void open_file_Slot();
	//    void quit_Slot();
	//    void save_file_Slot();
	//    void set_color_Slot();
};
}

#endif // PVIMPORTFILEDIALOG_H
