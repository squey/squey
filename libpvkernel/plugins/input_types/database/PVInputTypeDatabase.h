#ifndef PICVIZ_PVINPUTTYPEDATABASE_H
#define PICVIZ_PVINPUTTYPEDATABASE_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVInputType.h>

#include "../../common/database/PVDBQuery.h"

#include <QString>
#include <QStringList>

namespace PVRush {

class PVInputTypeDatabase: public PVInputTypeDesc<PVDBQuery>
{
public:
	PVInputTypeDatabase();
	virtual ~PVInputTypeDatabase();
public:
	bool createWidget(hash_formats const& formats, hash_formats& new_formats, list_inputs &inputs, QString& format, PVCore::PVArgumentList& args_ext, QWidget* parent = NULL) const;
	QString name() const;
	QString human_name() const;
	QString human_name_serialize() const;
	QString menu_input_name() const;
	QString tab_name_of_inputs(list_inputs const& in) const;
	QKeySequence menu_shortcut() const;
	bool get_custom_formats(PVInputDescription_p in, hash_formats &formats) const;

protected:
	mutable bool _is_custom_format;
	mutable PVFormat _custom_format;

	CLASS_REGISTRABLE_NOCOPY(PVInputTypeDatabase)
};

}

#endif
