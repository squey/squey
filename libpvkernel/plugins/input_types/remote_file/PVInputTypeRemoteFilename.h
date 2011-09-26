#ifndef PICVIZ_PVINPUTTYPEREMOTEFILENAME_H
#define PICVIZ_PVINPUTTYPEREMOTEFILENAME_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVInputType.h>

#include "../file/PVInputTypeFilename.h"

#include <QString>

namespace PVRush {

class PVInputTypeRemoteFilename: public PVInputTypeFilename
{
public:
	PVInputTypeRemoteFilename();
	virtual ~PVInputTypeRemoteFilename();
public:
	bool createWidget(hash_formats const& formats, list_inputs &inputs, QString& format, QWidget* parent = NULL) const;
	QString name() const;
	QString human_name() const;
	QString human_name_of_input(PVCore::PVArgument const& in) const;
	QString menu_input_name() const;
	QString tab_name_of_inputs(list_inputs const& in) const;
	QKeySequence menu_shortcut() const;
	bool get_custom_formats(PVCore::PVArgument const& in, hash_formats &formats) const;

protected:
	mutable QHash<QString, QUrl> _hash_real_filenames;
	
	CLASS_REGISTRABLE_NOCOPY(PVInputTypeRemoteFilename)
};

}

#endif
