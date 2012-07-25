/**
 * \file PVInputDescription.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVINPUTDESCRIPTION_FILE_H
#define PVINPUTDESCRIPTION_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSerializeArchive.h>

#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>

namespace PVRush {

class PVInputDescription: public boost::enable_shared_from_this<PVInputDescription>
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

}

#endif
