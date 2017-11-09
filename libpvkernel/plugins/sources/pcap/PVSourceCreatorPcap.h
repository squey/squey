/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2017
 */

#ifndef INENDI_PVSOURCECREATORPCAP_H
#define INENDI_PVSOURCECREATORPCAP_H

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVUnicodeSource.h>

namespace PVPcapsicum
{

class PVSourceCreatorPcap : public PVRush::PVSourceCreator
{
  public:
	source_p create_source_from_input(PVRush::PVInputDescription_p input) const override;
	QString supported_type() const override;
	PVRush::hash_formats get_supported_formats() const override;
	bool pre_discovery(PVRush::PVInputDescription_p input) const override;
	QString name() const override;

	CLASS_REGISTRABLE(PVSourceCreatorPcap)
};
} // namespace PVPcapsicum

#endif // INENDI_PVSOURCECREATORPCAP_H
