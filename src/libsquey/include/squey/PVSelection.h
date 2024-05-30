/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef SQUEY_PVSELECTION_H
#define SQUEY_PVSELECTION_H

#include <pvkernel/core/PVSelBitField.h> // for PVSelBitField

#include <pvbase/types.h> // for PVRow

#include <algorithm> // for move

namespace PVRush
{
class PVNraw;
} // namespace PVRush

namespace Squey
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
} // namespace Squey

#endif /* SQUEY_PVSELECTION_H */
