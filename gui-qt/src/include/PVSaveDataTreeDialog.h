/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVSAVEDATATREEDIALOG_H
#define PVSAVEDATATREEDIALOG_H

#include <pvkernel/widgets/PVFileDialog.h>

class QString;
class QCheckBox;
class QWidget;

namespace PVInspector
{

class PVSaveDataTreeDialog : public PVWidgets::PVFileDialog
{
  public:
	PVSaveDataTreeDialog(QString const& suffix, QString const& filter, QWidget* parent);

	bool save_log_file() const;

  protected:
	QCheckBox* _save_everything_checkbox;
};
} // namespace PVInspector

#endif
