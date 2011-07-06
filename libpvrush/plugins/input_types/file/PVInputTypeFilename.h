#ifndef PICVIZ_PVINPUTTYPEFILENAME_H
#define PICVIZ_PVINPUTTYPEFILENAME_H

#include <pvcore/general.h>
#include <pvrush/PVInputType.h>

#include <QString>
#include <QStringList>

namespace PVRush {

class LibExport PVInputTypeFilename: public PVInputType
{
public:
	virtual ~PVInputTypeFilename();
public:
	bool createWidget(hash_formats const& formats, list_inputs &inputs, QString& format, QWidget* parent = NULL) const;
	QString name() const;
	QString human_name() const;
	QString human_name_of_input(PVFilter::PVArgument const& in) const;
	QString menu_input_name() const;
	QString tab_name_of_inputs(list_inputs const& in) const;
	QKeySequence menu_shortcut() const;
	bool get_custom_formats(PVFilter::PVArgument const& in, hash_formats &formats) const;

protected:
	mutable QStringList _tmp_dir_to_delete;
	
	CLASS_REGISTRABLE(PVInputTypeFilename)
};

}

#endif
