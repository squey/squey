/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVINPUTDESCRIPTION_FILE_H
#define PVINPUTDESCRIPTION_FILE_H

#include <pvkernel/core/PVSerializeArchive.h>

namespace PVRush
{

class BadInputDescription : public std::runtime_error
{
  public:
	using std::runtime_error::runtime_error;
};

class PVInputDescription
{
	friend class PVCore::PVSerializeObject;

  public:
	typedef std::shared_ptr<PVInputDescription> p_type;

  public:
	virtual ~PVInputDescription() {}

  public:
	virtual bool operator==(const PVInputDescription& other) const = 0;
	bool operator!=(const PVInputDescription& other) const { return !operator==(other); }

  public:
	virtual QString human_name() const = 0;

  public:
	virtual void save_to_qsettings(QSettings& settings) const = 0;

  public:
	virtual void serialize_write(PVCore::PVSerializeObject& so) const = 0;
};

typedef PVInputDescription::p_type PVInputDescription_p;
} // namespace PVRush

#endif
