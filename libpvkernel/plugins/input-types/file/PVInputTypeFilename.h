/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVINPUTTYPEFILENAME_H
#define INENDI_PVINPUTTYPEFILENAME_H

#include "PVImportFileDialog.h"

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVFileDescription.h>

#include <QString>
#include <QStringList>
#include <QIcon>
#include <QCursor>

namespace PVRush
{

class PVInputTypeFilename : public PVInputTypeDesc<PVFileDescription>
{
  public:
	PVInputTypeFilename();
	virtual ~PVInputTypeFilename();

  public:
	bool createWidget(hash_formats const& formats,
	                  hash_formats& new_formats,
	                  list_inputs& inputs,
	                  QString& format,
	                  PVCore::PVArgumentList& args_ext,
	                  QWidget* parent = nullptr) const;
	QString name() const;
	QString human_name() const;
	QString human_name_serialize() const;
	QString internal_name() const;
	QString menu_input_name() const;
	QString tab_name_of_inputs(list_inputs const& in) const;
	QKeySequence menu_shortcut() const;
	bool get_custom_formats(PVInputDescription_p in, hash_formats& formats) const;

	QIcon icon() const { return QIcon(":/import-icon-white"); }
	QCursor cursor() const { return QCursor(Qt::PointingHandCursor); }

  protected:
	/**
	 * Push input corresponding to filenames in "inputs" performing uncompress if required.
	 */
	bool load_files(QStringList const& filenames, list_inputs& inputs, QWidget* parent) const;

  protected:
	mutable QStringList _tmp_dir_to_delete;
	int _limit_nfds;

  protected:
	CLASS_REGISTRABLE_NOCOPY(PVInputTypeFilename)
};
}

#endif
