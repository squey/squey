#ifndef PICVIZ_PVSOURCECREATORPCAPFILE_H
#define PICVIZ_PVSOURCECREATORPCAPFILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVUnicodeSource.h>

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
