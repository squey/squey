/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVSELECTION_H
#define INENDI_PVSELECTION_H

#include <pvbase/general.h>
#include <pvkernel/core/inendi_intrin.h>
#include <pvkernel/core/PVAllocators.h>
#include <pvkernel/core/PVBitVisitor.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/core/PVSelBitField.h>

#include <tbb/parallel_for.h>

#include <QTextStream>

#include <vector>

namespace PVRush
{
class PVNraw;
}

namespace Inendi
{

/**
* \class PVSelection
*/
class PVSelection : public PVCore::PVSelBitField
{
  public:
	explicit PVSelection(PVRow row_count) : PVCore::PVSelBitField(row_count) {}
	explicit PVSelection(PVCore::PVSelBitField&& bf) : PVCore::PVSelBitField(std::move(bf)) {}
	explicit PVSelection(PVCore::PVSelBitField const& bf) : PVCore::PVSelBitField(bf) {}

	PVSelection(PVSelection const& o) = default;
	PVSelection(PVSelection&& o) = default;

  public:
	inline PVSelection& operator|=(const PVSelection& rhs)
	{
		PVCore::PVSelBitField::operator|=(rhs);
		return *this;
	}

	PVSelection& operator=(const PVSelection& rhs) = default;
	PVSelection& operator=(PVSelection&& rhs) = default;

	PVSelection operator~() const { return PVSelection(PVCore::PVSelBitField::operator~()); }

	inline PVSelection& operator&=(const PVSelection& rhs)
	{
		PVCore::PVSelBitField::operator&=(rhs);
		return *this;
	};
	inline PVSelection& operator-=(const PVSelection& rhs)
	{
		PVCore::PVSelBitField::operator-=(rhs);
		return *this;
	};
	inline PVSelection& operator^=(const PVSelection& rhs)
	{
		PVCore::PVSelBitField::operator^=(rhs);
		return *this;
	};

	inline PVSelection operator&(const PVSelection& rhs) const
	{
		PVSelection ret(*this);
		ret &= rhs;
		return ret;
	}
	inline PVSelection operator-(const PVSelection& rhs) const
	{
		PVSelection ret(*this);
		ret -= rhs;
		return ret;
	}
	inline PVSelection operator^(const PVSelection& rhs) const
	{
		PVSelection ret(*this);
		ret ^= rhs;
		return ret;
	}
	inline PVSelection operator|(const PVSelection& rhs) const
	{
		PVSelection ret(*this);
		ret |= rhs;
		return ret;
	}
};
}

#endif /* INENDI_PVSELECTION_H */
