/**
 * \file PVSourceCreatorTexthdfs.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PICVIZ_PVSOURCECREATORTEXTHDFS_H
#define PICVIZ_PVSOURCECREATORTEXTHDFS_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVUnicodeSource.h>

namespace PVRush {

class PVSourceCreatorTexthdfs: public PVSourceCreator
{
public:
	source_p create_source_from_input(input_type input, PVFormat& used_format) const;
	source_p create_discovery_source_from_input(input_type input, const PVFormat& format) const;
	QString supported_type() const;
	hash_formats get_supported_formats() const;
	bool pre_discovery(input_type input) const;
	QString name() const;

	CLASS_REGISTRABLE(PVSourceCreatorTexthdfs)
};

}

#endif
