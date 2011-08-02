/*
 * $Id: PVField.h 3221 2011-06-30 11:45:19Z aguinet $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#ifndef PVFIELD_FILE_H
#define PVFIELD_FILE_H

#include <pvcore/general.h>
#include <pvcore/PVBufferSlice.h>
#include <pvcore/PVElement.h>

namespace PVCore {

class LibCoreDecl PVField : public PVBufferSlice {
public:
	PVField(PVElement const& parent, char* begin, char* end);
public:
	bool valid() const;
	void set_invalid();
	PVElement* elt_parent();
	void set_parent(PVElement const& parent);
	void deep_copy();
protected:
	bool _valid;
	PVElementData* _parent;
};

}

#endif
