/**
 * \file PVImportFileDialog.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVRUSH_PVIMPORTFILEDIALOG_H
#define PVRUSH_PVIMPORTFILEDIALOG_H

#include <QComboBox>
#include <QCheckBox>
#include <QLineEdit>
#include <QFileDialog>

namespace PVRush {

/**
 * \class PVImportFileDialog
 */
class PVImportFileDialog : public QFileDialog
{
    Q_OBJECT

    QComboBox *treat_as_combobox;

public:
	PVImportFileDialog(QStringList pluginslist, QWidget *parent = 0);
	bool save_inv_elts() const { return _check_save_inv_elts->isChecked(); }

	QStringList normalized_plugins_list;

	QStringList getFileNames(QString& treat_as);
	void        setDefaults();
	QCheckBox *activate_netflow_checkbox;
	QLineEdit *from_line_edit;
	QLineEdit *to_line_edit;
	bool _check_archives;
	QCheckBox* _check_archives_checkbox;
	QCheckBox* _check_save_inv_elts;
};
}

#endif // PVIMPORTFILEDIALOG_H

