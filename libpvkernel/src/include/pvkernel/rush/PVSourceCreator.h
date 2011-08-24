#ifndef PICVIZ_PVSOURCECREATOR_H
#define PICVIZ_PVSOURCECREATOR_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVRegistrableClass.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/rush/PVRawSourceBase_types.h>
#include <pvkernel/rush/PVFormat.h>

#include <boost/shared_ptr.hpp>
#include <list>

namespace PVRush {

class LibKernelDecl PVSourceCreator: public PVCore::PVRegistrableClass< PVSourceCreator >
{
public:
	typedef PVRush::PVRawSourceBase source_t;
	typedef boost::shared_ptr<source_t> source_p;
	typedef boost::shared_ptr<PVSourceCreator> p_type;
public:
	virtual ~PVSourceCreator() {}
public:
	virtual source_p create_source_from_input(PVCore::PVArgument const& input, PVFormat& /*used_format*/) const { return create_discovery_source_from_input(input); }
	virtual source_p create_discovery_source_from_input(PVCore::PVArgument const& input) const = 0;
	virtual QString supported_type() const = 0;
	virtual QString name() const = 0;
	virtual hash_formats get_supported_formats() const = 0;

	// "pre-discovery" is called before processing the source into the TBB filters created
	// by its PVFormat objects. If this function returns false, this PVSourceCreator is automatically
	// discared (for *all* its formats) for this input.
	// FIXME: cool, but not used yet in PVMainWindow
	virtual bool pre_discovery(PVCore::PVArgument const& input) const = 0;
};

typedef PVSourceCreator::p_type PVSourceCreator_p;

}

#ifdef WIN32
LibKernelDeclExplicitTempl PVCore::PVClassLibrary<PVRush::PVSourceCreator>;
#endif

#endif
