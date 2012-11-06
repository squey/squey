/**
 * \file PVSelection.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PICVIZ_PVSELECTION_H
#define PICVIZ_PVSELECTION_H

#include <pvkernel/core/stdint.h>
#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/PVAllocators.h>
#include <pvkernel/core/PVBitVisitor.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/core/PVAlignedBlockedRange.h>
#include <pvkernel/core/PVSelBitField.h>

#include <picviz/general.h>

#include <tbb/parallel_for.h>

#include <QTextStream>

#include <vector>

namespace PVRush {
class PVNraw;
}

namespace Picviz {

class PVSparseSelection;

/**
* \class PVSelection
*/
class LibPicvizDecl PVSelection: public PVCore::PVSelBitField
{
	friend class PVCore::PVSerializeObject;

public:
	struct tag_allocate_empty { };

public:
	PVSelection(): PVCore::PVSelBitField() { }
	PVSelection(tag_allocate_empty):
		PVCore::PVSelBitField()
	{
		allocate_table();
		select_none();
	}

	PVSelection(PVSelection const& o): PVCore::PVSelBitField(o) { }
	PVSelection(PVSelection&& o): PVCore::PVSelBitField(o) { }


public:
	PVSelection& operator|=(const PVSparseSelection &rhs);

	inline PVSelection& operator|=(const PVSelection& rhs) { PVCore::PVSelBitField::operator|=(rhs); return *this; }
	inline PVSelection& operator= (const PVSelection &rhs) { PVCore::PVSelBitField::operator=(rhs); return *this; }
	inline PVSelection& operator= (PVSelection&& rhs) { PVCore::PVSelBitField::operator=(rhs); return *this; };
	inline PVSelection& operator&=(const PVSelection &rhs) { PVCore::PVSelBitField::operator&=(rhs); return *this; };
	inline PVSelection& operator-=(const PVSelection &rhs) { PVCore::PVSelBitField::operator-=(rhs); return *this; };
	inline PVSelection& operator^=(const PVSelection &rhs) { PVCore::PVSelBitField::operator^=(rhs); return *this; };

	inline PVSelection operator&(const PVSelection &rhs) const
	{
		PVSelection ret(*this);
		ret &= rhs;
		return std::move(ret);
	}
	inline PVSelection operator-(const PVSelection &rhs) const
	{
		PVSelection ret(*this);
		ret -= rhs;
		return std::move(ret);
	}
	inline PVSelection operator^(const PVSelection &rhs) const
	{
		PVSelection ret(*this);
		ret ^= rhs;
		return std::move(ret);
	}
	inline PVSelection operator~() const
	{
		PVSelection ret;
		move_from_base(ret, PVCore::PVSelBitField::operator~());
		return std::move(ret);
	}
	inline PVSelection operator|(const PVSelection &rhs) const
	{
		PVSelection ret(*this);
		ret |= rhs;
		return std::move(ret);
	}

	void write_selected_lines_nraw(QTextStream& stream, PVRush::PVNraw const& nraw, PVRow write_max);

private:
	static void move_from_base(PVSelection& ret, PVCore::PVSelBitField&& b)
	{
		assert(&ret != &b);
		ret._table = b._table;
		b._table = nullptr;
	}
};

}

#endif /* PICVIZ_PVSELECTION_H */
