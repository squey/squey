/**
 * \file PVInputTypeRemoteFilename.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PICVIZ_PVINPUTTYPEREMOTEFILENAME_H
#define PICVIZ_PVINPUTTYPEREMOTEFILENAME_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVInputType.h>

#include "../file/PVInputTypeFilename.h"

#include <QString>
#include <QIcon>
#include <QCursor>

namespace PVRush {

class PVInputTypeRemoteFilename: public PVInputTypeFilename
{
public:
	PVInputTypeRemoteFilename();
	virtual ~PVInputTypeRemoteFilename();
public:
	bool createWidget(hash_formats const& formats, hash_formats& new_formats, list_inputs &inputs, QString& format, PVCore::PVArgumentList& args_ext, QWidget* parent = NULL) const;
	QString name() const;
	QString human_name() const;
	QString human_name_serialize() const;
	QString human_name_of_input(PVInputDescription_p in) const;
	QString menu_input_name() const;
	QString tab_name_of_inputs(list_inputs const& in) const;
	QKeySequence menu_shortcut() const;
	bool get_custom_formats(PVInputDescription_p in, hash_formats &formats) const;

	QIcon icon() const { return QIcon(":/import-icon-white"); }
	QCursor cursor() const { return QCursor(Qt::PointingHandCursor); }

protected:
	mutable QHash<QString, QUrl> _hash_real_filenames;

	CLASS_REGISTRABLE_NOCOPY(PVInputTypeRemoteFilename)
};

}

#endif
