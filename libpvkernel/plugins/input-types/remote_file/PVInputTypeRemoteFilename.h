/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVINPUTTYPEREMOTEFILENAME_H
#define INENDI_PVINPUTTYPEREMOTEFILENAME_H

#include <pvkernel/rush/PVInputType.h>

#include "../file/PVInputTypeFilename.h"

#include <QString>
#include <QIcon>
#include <QCursor>

namespace PVRush
{

class PVInputTypeRemoteFilename : public PVInputTypeFilename
{
  public:
	PVInputTypeRemoteFilename();

  public:
	bool createWidget(hash_formats& formats,
	                  list_inputs& inputs,
	                  QString& format,
	                  PVCore::PVArgumentList& args_ext,
	                  QWidget* parent = nullptr) const override;
	QString name() const override;
	QString human_name() const override;
	QString human_name_serialize() const override;
	QString internal_name() const override;
	QString human_name_of_input(PVInputDescription_p in) const override;
	QString menu_input_name() const override;
	QString tab_name_of_inputs(list_inputs const& in) const override;
	QKeySequence menu_shortcut() const override;
	bool get_custom_formats(PVInputDescription_p in, hash_formats& formats) const override;

	QIcon icon() const override { return QIcon(":/import-icon-white"); }
	QCursor cursor() const override { return QCursor(Qt::PointingHandCursor); }

  protected:
	mutable QHash<QString, QUrl> _hash_real_filenames;

	CLASS_REGISTRABLE_NOCOPY(PVInputTypeRemoteFilename)
};
} // namespace PVRush

#endif
