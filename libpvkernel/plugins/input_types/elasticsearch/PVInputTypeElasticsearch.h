/**
 * \file PVInputTypeElasticsearch.h
 *
 * Copyright (C) Picviz Labs 2015
 */

#ifndef PICVIZ_PVINPUTTYPEDATABASE_H
#define PICVIZ_PVINPUTTYPEDATABASE_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVInputType.h>

#include "../../common/elasticsearch/PVElasticsearchQuery.h"

#include <QString>
#include <QStringList>

namespace PVRush {

class PVInputTypeElasticsearch: public PVInputTypeDesc<PVElasticsearchQuery>
{
public:
	PVInputTypeElasticsearch();
	virtual ~PVInputTypeElasticsearch();
public:
	bool createWidget(hash_formats const& formats, hash_formats& new_formats, list_inputs &inputs, QString& format, PVCore::PVArgumentList& args_ext, QWidget* parent = NULL) const;
	QString name() const;
	QString human_name() const;
	QString human_name_serialize() const;
	QString internal_name() const override;
	QString menu_input_name() const;
	QString tab_name_of_inputs(list_inputs const& in) const;
	QKeySequence menu_shortcut() const;
	bool get_custom_formats(PVInputDescription_p in, hash_formats &formats) const;

	QIcon icon() const { return QIcon(":/import-icon-white"); }
	QCursor cursor() const { return QCursor(Qt::PointingHandCursor); }

protected:
	mutable bool _is_custom_format;
	mutable PVFormat _custom_format;

	CLASS_REGISTRABLE_NOCOPY(PVInputTypeElasticsearch)
};

}

#endif
