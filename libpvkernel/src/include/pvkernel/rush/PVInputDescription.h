/**
 * \file PVInputDescription.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVINPUTDESCRIPTION_FILE_H
#define PVINPUTDESCRIPTION_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSerializeArchive.h>

#include <memory>

namespace PVRush {

class PVInputDescription: public std::enable_shared_from_this<PVInputDescription>
{
	friend class PVCore::PVSerializeObject;
public:
	typedef std::shared_ptr<PVInputDescription> p_type;

public:
	virtual ~PVInputDescription() { }

public:
	virtual bool operator==(const PVInputDescription& other) const = 0;
	bool operator!=(const PVInputDescription& other) const { return ! operator==(other); }

public:
	virtual QString human_name() const = 0;

public:
	virtual void save_to_qsettings(QSettings& settings) const = 0;
	virtual void load_from_qsettings(const QSettings& settings) = 0;

protected:
	virtual void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v) = 0;
};

typedef PVInputDescription::p_type PVInputDescription_p;

}

#endif
