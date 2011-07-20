#ifndef PICVIZ_PVSOURCECREATORTEXTFILE_H
#define PICVIZ_PVSOURCECREATORTEXTFILE_H

#include <pvcore/general.h>
#include <pvcore/PVArgument.h>
#include <pvrush/PVSourceCreator.h>
#include <pvrush/PVUnicodeSource.h>

namespace PVRush {

class PVSourceCreatorTextfile: public PVSourceCreator
{
public:
	source_p create_source_from_input(PVCore::PVArgument const& input) const;
	QString supported_type() const;
	hash_formats get_supported_formats() const;
	bool pre_discovery(PVCore::PVArgument const& input) const;
	QString name() const;

	CLASS_REGISTRABLE(PVSourceCreatorTextfile)
};

}

#endif
