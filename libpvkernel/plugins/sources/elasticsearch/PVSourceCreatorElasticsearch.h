/**
 * \file PVSourceCreatorElasticsearch.h
 *
 * Copyright (C) Picviz Labs 2015
 */

#ifndef PICVIZ_PVSOURCECREATORELASTICSEARCH_H
#define PICVIZ_PVSOURCECREATORELASTICSEARCH_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/rush/PVSourceCreator.h>

namespace PVRush {

class PVSourceCreatorElasticsearch: public PVSourceCreator
{
public:
	source_p create_discovery_source_from_input(PVInputDescription_p input, const PVFormat& format) const;
	QString supported_type() const;
	hash_formats get_supported_formats() const;
	bool pre_discovery(PVInputDescription_p input) const;
	QString name() const;

	CLASS_REGISTRABLE(PVSourceCreatorElasticsearch)
};

}

#endif	// PICVIZ_PVSOURCECREATORELASTICSEARCH_H
