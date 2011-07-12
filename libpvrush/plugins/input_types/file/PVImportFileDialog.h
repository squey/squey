//! \file PVImportFileDialog.h
//! $Id: PVImportFileDialog.h 2714 2011-05-12 08:02:27Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

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

	QStringList normalized_plugins_list;

	QStringList getFileNames(QString& treat_as);
	void        setDefaults();
	QCheckBox *activate_netflow_checkbox;
	QLineEdit *from_line_edit;
	QLineEdit *to_line_edit;
	bool _check_archives;
	QCheckBox* _check_archives_checkbox;
};
}

#endif // PVIMPORTFILEDIALOG_H

