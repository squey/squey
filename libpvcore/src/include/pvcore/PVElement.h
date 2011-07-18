/*
 * $Id: PVElement.h 3221 2011-06-30 11:45:19Z aguinet $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#ifndef PVELEMENT_FILE_H
#define PVELEMENT_FILE_H

#include <pvcore/general.h>
#include <pvcore/PVBufferSlice.h>

#include <list>

#include <tbb/scalable_allocator.h>
#include <tbb/tbb_allocator.h>

#include <QList>
#include <QExplicitlySharedDataPointer>

namespace PVCore {

class PVChunk;

class PVField;
class PVElementData;

typedef std::list<PVField, tbb::tbb_allocator<PVField> > list_fields;
class LibCoreDecl PVElement : public PVBufferSlice {
	friend class PVField;
public:
	PVElement(PVChunk* parent, char* begin, char* end);
	PVElement(PVElement const& src);
	virtual ~PVElement();
public:
	bool valid() const;
	void set_invalid();
	void set_parent(PVChunk* parent);
	PVChunk* chunk_parent();

	list_fields& fields();
	list_fields const& c_fields() const;

	buf_list_t& realloc_bufs() const;
public:
	PVElement& operator=(PVElement const& src);
protected:
	QExplicitlySharedDataPointer<PVElementData> d;
};

}

#endif
