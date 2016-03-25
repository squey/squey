/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVRUSH_PVIMPORTFILEDIALOG_H
#define PVRUSH_PVIMPORTFILEDIALOG_H

#include <QFileDialog>

class QComboBox;


namespace PVRush {

/**
 * \class PVImportFileDialog
 *
 * Speciale FileDialog with option to choose the format.
 */
class PVImportFileDialog : public QFileDialog
{
    Q_OBJECT

public:
	PVImportFileDialog(QStringList pluginslist, QWidget *parent = 0);

	/**
	 * Open the FileDialog and return list of selected file with selected format.
	 */
	QStringList getFileNames(QString& treat_as);

private:
	QComboBox *treat_as_combobox; //!< Combo box with formats.

};
}

#endif // PVIMPORTFILEDIALOG_H

