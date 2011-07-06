//! \file PVImportFileDialog.h
//! $Id: PVImportFileDialog.h 2714 2011-05-12 08:02:27Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVIMPORTFILEDIALOG_H
#define PVIMPORTFILEDIALOG_H

#include <QComboBox>
#include <QCheckBox>
#include <QLineEdit>
#include <QFileDialog>

#include <picviz/general.h>
#include <picviz/plugins.h>

namespace PVInspector {

/**
 * \class PVImportFileDialog
 */
class PVImportFileDialog : public QFileDialog
{
    Q_OBJECT

    QComboBox *treat_as_combobox;

public:
	PVImportFileDialog(QWidget *parent = 0);

	QStringList normalized_plugins_list;

	QStringList getFileNames(QString& treat_as);
	void        setDefaults();
	QCheckBox *activate_netflow_checkbox;
	QLineEdit *from_line_edit;
	QLineEdit *to_line_edit;
//public slots:
//    void new_file_Slot();
//    void new_scene_Slot();
//    void open_file_Slot();
//    void quit_Slot();
//    void save_file_Slot();
//    void select_scene_Slot();
//    void set_color_Slot();
};
}

#endif // PVIMPORTFILEDIALOG_H

