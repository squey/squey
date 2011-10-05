#ifndef PVINPUTDESCRIPTION_FILE_H
#define PVINPUTDESCRIPTION_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSerializeArchive.h>

#include <boost/shared_ptr.hpp>

namespace PVRush {

class PVInputDescription
{
	friend class PVCore::PVSerializeObject;
public:
	typedef boost::shared_ptr<PVInputDescription> p_type;

public:
	virtual ~PVInputDescription() { }

public:
	virtual QString human_name() const = 0;

protected:
	virtual void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v) = 0;
};

typedef PVInputDescription::p_type PVInputDescription_p;
typedef PVInputDescription_p input_type;

}

#endif
