/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVSELECTION_H
#define INENDI_PVSELECTION_H

#include <pvkernel/core/PVSelBitField.h> // for PVSelBitField

#include <pvbase/types.h> // for PVRow

#include <algorithm> // for move

namespace PVRush
{
class PVNraw;
} // namespace PVRush

namespace Inendi
{

/**
* \class PVSelection
*/
class PVSelection : public PVCore::PVSelBitField
{
  public:
	template <class T>
	explicit PVSelection(T&& t) : PVCore::PVSelBitField(std::forward<T>(t))
	{
	}

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

	PVSelection operator~() const { return PVSelection{PVCore::PVSelBitField::operator~()}; }

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
} // namespace Inendi

#endif /* INENDI_PVSELECTION_H */
