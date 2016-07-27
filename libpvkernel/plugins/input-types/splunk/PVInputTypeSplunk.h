/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVINPUTTYPESPLUNK_H
#define INENDI_PVINPUTTYPESPLUNK_H

#include <pvkernel/rush/PVInputType.h>

#include "../../common/splunk/PVSplunkInfos.h"
#include "../../common/splunk/PVSplunkQuery.h"

#include <QString>
#include <QStringList>

namespace PVRush
{

class PVInputTypeSplunk : public PVInputTypeDesc<PVSplunkQuery>
{
  public:
	PVInputTypeSplunk();
	virtual ~PVInputTypeSplunk();

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
	QString internal_name() const override;
	QString menu_input_name() const;
	QString tab_name_of_inputs(list_inputs const& in) const;
	QKeySequence menu_shortcut() const;
	bool get_custom_formats(PVInputDescription_p in, hash_formats& formats) const;

	QIcon icon() const { return QIcon(":/splunk_icon"); }
	QCursor cursor() const { return QCursor(Qt::PointingHandCursor); }

  public:
	PVInputDescription_p serialize_read(PVCore::PVSerializeObject& so) override
	{
		// FIXME : Should be improve to not use default constructor.
		auto query = new PVSplunkQuery();
		query->serialize_read(so);
		return PVInputDescription_p(query);
	}

  protected:
	mutable bool _is_custom_format;
	mutable PVFormat _custom_format;

	CLASS_REGISTRABLE_NOCOPY(PVInputTypeSplunk)
};

} // namespace PVRush

#endif
