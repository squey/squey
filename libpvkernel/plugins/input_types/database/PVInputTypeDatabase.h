#ifndef PICVIZ_PVINPUTTYPEDATABASE_H
#define PICVIZ_PVINPUTTYPEDATABASE_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVInputType.h>

#include <QString>
#include <QStringList>

namespace PVRush {

class PVInputTypeDatabase: public PVInputType
{
public:
	PVInputTypeDatabase();
	virtual ~PVInputTypeDatabase();
public:
	bool createWidget(hash_formats const& formats, list_inputs &inputs, QString& format, QWidget* parent = NULL) const;
	QString name() const;
	QString human_name() const;
	QString human_name_of_input(PVCore::PVArgument const& in) const;
	QString menu_input_name() const;
	QString tab_name_of_inputs(list_inputs const& in) const;
	QKeySequence menu_shortcut() const;
	bool get_custom_formats(PVCore::PVArgument const& in, hash_formats &formats) const;

	CLASS_REGISTRABLE_NOCOPY(PVInputTypeDatabase)
};

}

#endif
