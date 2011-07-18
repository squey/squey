#ifndef PICVIZ_PVSOURCECREATOR_H
#define PICVIZ_PVSOURCECREATOR_H

#include <pvcore/general.h>
#include <pvcore/PVRegistrableClass.h>
#include <pvcore/PVClassLibrary.h>
#include <pvfilter/PVArgument.h>
#include <pvfilter/PVRawSourceBase.h>
#include <pvrush/PVFormat.h>

#include <boost/shared_ptr.hpp>
#include <list>

namespace PVRush {

class LibRushDecl PVSourceCreator: public PVCore::PVRegistrableClass< PVSourceCreator >
{
public:
	typedef PVFilter::PVRawSourceBase source_t;
	typedef boost::shared_ptr<source_t> source_p;
	typedef boost::shared_ptr<PVSourceCreator> p_type;
public:
	virtual ~PVSourceCreator() {}
public:
	virtual source_p create_source_from_input(PVFilter::PVArgument const& input) const = 0;
	virtual QString supported_type() const = 0;
	virtual QString name() const = 0;
	virtual hash_formats get_supported_formats() const = 0;

	// "pre-discovery" is called before processing the source into the TBB filters created
	// by its PVFormat objects. If this function returns false, this PVSourceCreator is automatically
	// discared (for *all* its formats) for this input.
	// FIXME: cool, but not used yet in PVMainWindow
	virtual bool pre_discovery(PVFilter::PVArgument const& input) const = 0;
};

typedef PVSourceCreator::p_type PVSourceCreator_p;

}

#ifdef WIN32
pvrush_FilterLibraryDecl PVCore::PVClassLibrary<PVRush::PVSourceCreator>;
#endif

#endif
