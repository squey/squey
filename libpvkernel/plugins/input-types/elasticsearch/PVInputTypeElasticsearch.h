/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVINPUTTYPEELASTICSEARCH_H
#define INENDI_PVINPUTTYPEELASTICSEARCH_H

#include <pvkernel/rush/PVInputType.h>

#include "../../common/elasticsearch/PVElasticsearchQuery.h"

#include <QString>
#include <QStringList>

namespace PVRush
{

class PVInputTypeElasticsearch : public PVInputTypeDesc<PVElasticsearchQuery>
{
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
	QString menu_input_name() const override;
	QString tab_name_of_inputs(list_inputs const& in) const override;
	QKeySequence menu_shortcut() const override;
	bool get_custom_formats(PVInputDescription_p in, hash_formats& formats) const override;

	QIcon icon() const override { return QIcon(":/elasticsearch_icon"); }
	QCursor cursor() const override { return QCursor(Qt::PointingHandCursor); }

	CLASS_REGISTRABLE_NOCOPY(PVInputTypeElasticsearch)
};
} // namespace PVRush

#endif
