/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

#ifndef INENDI_PVINPUTTYPEELASTICSEARCH_H
#define INENDI_PVINPUTTYPEELASTICSEARCH_H

#include <pvkernel/rush/PVInputType.h>

#include "../../common/opcua/PVOpcUaQuery.h"

#include <QString>
#include <QStringList>

namespace PVRush
{

class PVInputTypeOpcUa : public PVInputTypeDesc<PVOpcUaQuery>
{
  public:
	bool createWidget(hash_formats const& formats,
	                  hash_formats& new_formats,
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

	QIcon icon() const override { return QIcon(":/opcua_icon"); }
	QCursor cursor() const override { return QCursor(Qt::PointingHandCursor); }

	CLASS_REGISTRABLE_NOCOPY(PVInputTypeOpcUa)
};
} // namespace PVRush

#endif