/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVSaveDataTreeDialog.h"

#include <pvbase/general.h>

#include <QGridLayout>
#include <QCheckBox>

PVInspector::PVSaveDataTreeDialog::PVSaveDataTreeDialog(QString const& suffix,
                                                        QString const& filter,
                                                        QWidget* parent)
    : QFileDialog(parent)
{
	// Do not use native dialog as we modify the layout
	setOption(QFileDialog::DontUseNativeDialog, true);
	setAcceptMode(QFileDialog::AcceptSave);
	setDefaultSuffix(suffix);
	setWindowTitle(tr("Save project..."));
	setNameFilters(QStringList() << filter << ALL_FILES_FILTER);

	QGridLayout* main_layout = (QGridLayout*)layout();

	_save_everything_checkbox = new QCheckBox(tr("Include original files"), this);
	_save_everything_checkbox->setTristate(false);
	main_layout->addWidget(_save_everything_checkbox, 5, 1);
}

bool PVInspector::PVSaveDataTreeDialog::save_log_file() const
{
	return _save_everything_checkbox->isChecked();
}
