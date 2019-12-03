/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVSOURCECREATOR_H
#define INENDI_PVSOURCECREATOR_H

#include <pvkernel/core/PVRegistrableClass.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVInputType.h>

#include <memory>

namespace PVRush
{

class PVRawSourceBase;

class PVSourceCreator : public PVCore::PVRegistrableClass<PVSourceCreator>
{
  public:
	typedef PVRush::PVRawSourceBase source_t;
	typedef std::shared_ptr<source_t> source_p;
	typedef std::shared_ptr<PVSourceCreator> p_type;

  public:
	virtual ~PVSourceCreator() = default;

  public:
	virtual source_p create_source_from_input(PVInputDescription_p input) const = 0;
	virtual QString supported_type() const = 0;
	PVInputType_p supported_type_lib()
	{
		QString name = supported_type();
		PVInputType_p type_lib = LIB_CLASS(PVInputType)::get().get_class_by_name(name);
		return type_lib->clone<PVInputType>();
	}
	virtual QString name() const = 0;
	virtual bool custom_multi_inputs() const { return false; }
};

typedef PVSourceCreator::p_type PVSourceCreator_p;
} // namespace PVRush

#endif
