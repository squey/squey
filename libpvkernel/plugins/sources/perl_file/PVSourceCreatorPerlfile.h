/**
 * \file PVSourceCreatorPerlfile.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PICVIZ_PVSOURCECREATORPERLFILE_H
#define PICVIZ_PVSOURCECREATORPERLFILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/rush/PVSourceCreator.h>

#include "PVPerlSource.h"


namespace PVRush {

class PVSourceCreatorPerlfile: public PVSourceCreator
{
public:
	source_p create_discovery_source_from_input(PVInputDescription_p input, const PVFormat& format) const;
	QString supported_type() const;
	hash_formats get_supported_formats() const;
	bool pre_discovery(PVInputDescription_p input) const;
	QString name() const;

	CLASS_REGISTRABLE(PVSourceCreatorPerlfile)
};

}

#endif	/* PICVIZ_PVSOURCECREATORPERLFILE_H */
