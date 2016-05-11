/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVSOURCECREATORELASTICSEARCH_H
#define INENDI_PVSOURCECREATORELASTICSEARCH_H

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/rush/PVSourceCreator.h>

namespace PVRush
{

class PVSourceCreatorElasticsearch : public PVSourceCreator
{
  public:
	source_p create_source_from_input(PVInputDescription_p input, const PVFormat& format) const;
	QString supported_type() const;
	hash_formats get_supported_formats() const;
	bool pre_discovery(PVInputDescription_p input) const;
	QString name() const;

	CLASS_REGISTRABLE(PVSourceCreatorElasticsearch)
};
}

#endif // INENDI_PVSOURCECREATORELASTICSEARCH_H
