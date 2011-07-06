#ifndef PICVIZ_PVINPUTTYPE_H
#define PICVIZ_PVINPUTTYPE_H

#include <pvcore/general.h>
#include <pvcore/PVRegistrableClass.h>
#include <pvcore/PVClassLibrary.h>
#include <pvfilter/PVArgument.h>
#include <pvrush/PVFormat.h>
#include <QList>
#include <QKeySequence>

namespace PVRush {

class LibExport PVInputType: public PVCore::PVRegistrableClass<PVInputType>
{
public:
	typedef PVFilter::PVArgument input_type;
	typedef QList<input_type> list_inputs;
	typedef boost::shared_ptr<PVInputType> p_type;
public:
	virtual bool createWidget(hash_formats const& formats, list_inputs &inputs, QString& format, QWidget* parent = NULL) const = 0;
	virtual QString name() const = 0;
	virtual QString human_name() const = 0;
	// Warning: the "human name" of an input must be *unique* accross all the possible inputs
	virtual QString human_name_of_input(input_type const& in) const = 0;
	virtual QString menu_input_name() const = 0;
	virtual QKeySequence menu_shortcut() const { return QKeySequence(); }
	virtual QString tab_name_of_inputs(list_inputs const& in) const = 0;
	virtual bool get_custom_formats(input_type const& in, hash_formats &formats) const = 0;
public:
	QStringList human_name_of_inputs(list_inputs const& in) const
	{
		QStringList ret;
		list_inputs::const_iterator it;
		for (it = in.begin(); it != in.end(); it++) {
			ret << human_name_of_input(*it);
		}
		return ret;
	}
};

typedef PVInputType::p_type PVInputType_p;

}

//#define REGISTER_INPUT_TYPE(T) REGISTER_CLASS(T().name(), T())
#ifdef WIN32
pvrush_FilterLibraryDecl PVCore::PVClassLibrary<PVRush::PVInputType>;
#endif

#endif
