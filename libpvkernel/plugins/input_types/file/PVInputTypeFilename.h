#ifndef PICVIZ_PVINPUTTYPEFILENAME_H
#define PICVIZ_PVINPUTTYPEFILENAME_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVInputType.h>

#include <QString>
#include <QStringList>

namespace PVRush {

class PVInputTypeFilename: public PVInputType
{
public:
	PVInputTypeFilename();
	virtual ~PVInputTypeFilename();
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
	mutable QStringList _tmp_dir_to_delete;
	int _limit_nfds;
	
	CLASS_REGISTRABLE_NOCOPY(PVInputTypeFilename)
};

}

#endif
