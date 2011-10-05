#ifndef PICVIZ_PVINPUTTYPEHDFS_H
#define PICVIZ_PVINPUTTYPEHDFS_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVInputType.h>

#include <QString>
#include <QStringList>

namespace PVRush {

class PVInputTypeHDFS: public PVInputType
{
public:
	PVInputTypeHDFS();
	virtual ~PVInputTypeHDFS();
public:
	bool createWidget(hash_formats const& formats, hash_formats& new_formats, list_inputs &inputs, QString& format, QWidget* parent = NULL) const;
	QString name() const;
	QString human_name() const;
	QString menu_input_name() const;
	QString tab_name_of_inputs(list_inputs const& in) const;
	QKeySequence menu_shortcut() const;
	bool get_custom_formats(input_type in, hash_formats &formats) const;

protected:
	mutable QStringList _tmp_dir_to_delete;
	
	CLASS_REGISTRABLE_NOCOPY(PVInputTypeHDFS)
};

}

#endif
