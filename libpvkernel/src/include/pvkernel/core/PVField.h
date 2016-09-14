/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFIELD_FILE_H
#define PVFIELD_FILE_H

#include <pvkernel/core/PVBufferSlice.h>

namespace PVCore
{

class PVElement;

class PVField : public PVBufferSlice
{
  public:
	PVField(PVElement& parent, char* begin, char* end);
	PVField(PVElement& parent);

  public:
	bool valid() const;
	void set_invalid();
	bool filtered() const;
	void set_filtered();
	PVElement* elt_parent();

  private:
	void init(PVElement& parent);

  protected:
	bool _valid;
	bool _filtered;
	PVElement* _parent;
};
} // namespace PVCore

#endif
