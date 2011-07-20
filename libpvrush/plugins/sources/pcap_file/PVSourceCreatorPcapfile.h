#ifndef PICVIZ_PVSOURCECREATORPCAPFILE_H
#define PICVIZ_PVSOURCECREATORPCAPFILE_H

#include <pvcore/general.h>
#include <pvcore/PVArgument.h>
#include <pvrush/PVSourceCreator.h>
#include <pvrush/PVUnicodeSource.h>

namespace PVRush {

class PVSourceCreatorPcapfile: public PVSourceCreator
{
public:
	source_p create_source_from_input(PVCore::PVArgument const& input) const;
	QString supported_type() const;
	hash_formats get_supported_formats() const;
	bool pre_discovery(PVCore::PVArgument const& input) const;
	QString name() const;

	CLASS_REGISTRABLE(PVSourceCreatorPcapfile)
};

}

#endif
