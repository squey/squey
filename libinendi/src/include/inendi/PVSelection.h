/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVSELECTION_H
#define INENDI_PVSELECTION_H

#include <pvkernel/core/inendi_intrin.h>
#include <pvkernel/core/PVAllocators.h>
#include <pvkernel/core/PVBitVisitor.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/core/PVAlignedBlockedRange.h>
#include <pvkernel/core/PVSelBitField.h>

#include <inendi/general.h>

#include <tbb/parallel_for.h>

#include <QTextStream>

#include <vector>

namespace PVRush {
class PVNraw;
}

namespace Inendi {

class PVAxesCombination;
class PVSparseSelection;

/**
* \class PVSelection
*/
class PVSelection: public PVCore::PVSelBitField
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
	inline bool operator==(const PVSelection &rhs) const { return PVCore::PVSelBitField::operator==(rhs); }

	inline PVSelection operator&(const PVSelection &rhs) const
	{
		PVSelection ret(*this);
		ret &= rhs;
		return ret;
	}
	inline PVSelection operator-(const PVSelection &rhs) const
	{
		PVSelection ret(*this);
		ret -= rhs;
		return ret;
	}
	inline PVSelection operator^(const PVSelection &rhs) const
	{
		PVSelection ret(*this);
		ret ^= rhs;
		return ret;
	}
	inline PVSelection operator~() const
	{
		PVSelection ret;
		move_from_base(ret, PVCore::PVSelBitField::operator~());
		return ret;
	}
	inline PVSelection operator|(const PVSelection &rhs) const
	{
		PVSelection ret(*this);
		ret |= rhs;
		return ret;
	}

private:
	static void move_from_base(PVSelection& ret, PVCore::PVSelBitField&& b)
	{
		assert(&ret != &b);
		if (ret._table) {
			ret.free_table();
		}
		ret._table = b._table;
		b._table = nullptr;
	}
};

}

#endif /* INENDI_PVSELECTION_H */