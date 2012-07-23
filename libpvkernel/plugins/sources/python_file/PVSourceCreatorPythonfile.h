/**
 * \file PVSourceCreatorPythonfile.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PICVIZ_PVSOURCECREATORPYTHONFILE_H
#define PICVIZ_PVSOURCECREATORPYTHONFILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/rush/PVSourceCreator.h>

namespace PVRush {

class PVSourceCreatorPythonfile: public PVSourceCreator
{
public:
	source_p create_discovery_source_from_input(input_type input, const PVFormat& format) const;
	QString supported_type() const;
	hash_formats get_supported_formats() const;
	bool pre_discovery(input_type input) const;
	QString name() const;

	CLASS_REGISTRABLE(PVSourceCreatorPythonfile)
};

}

#endif	/* PICVIZ_PVSOURCECREATORPYTHONFILE_H */
