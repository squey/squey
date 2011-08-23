#ifndef PICVIZ_PVSOURCECREATORDATABASE_H
#define PICVIZ_PVSOURCECREATORDATABASE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/rush/PVSourceCreator.h>

namespace PVRush {

class PVSourceCreatorDatabase: public PVSourceCreator
{
public:
	source_p create_discovery_source_from_input(PVCore::PVArgument const& input) const;
	QString supported_type() const;
	hash_formats get_supported_formats() const;
	bool pre_discovery(PVCore::PVArgument const& input) const;
	QString name() const;

	CLASS_REGISTRABLE(PVSourceCreatorDatabase)
};

}

#endif
