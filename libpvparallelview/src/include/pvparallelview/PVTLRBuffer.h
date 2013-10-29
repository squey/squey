
#ifndef PARALLELVIEW_PVTLRBUFFER_H
#define PARALLELVIEW_PVTLRBUFFER_H

#include <pvparallelview/common.h>

namespace PVParallelView
{

// #define TLR_USE_C_BITFIELD

template <size_t Bbits=PARALLELVIEW_ZZT_BBITS>
class PVTLRBuffer
{
public:
	constexpr static size_t length = 3 * (1 << (2*Bbits));

public:
	struct index_t
	{
		index_t(uint32_t vv = 0U)
		{
			v = vv;
		}

		index_t(uint32_t t, uint32_t l, uint32_t r)
		{
#ifdef TLR_USE_C_BITFIELD
			v = 0U;
			s.t = t;
			s.l = l;
			s.r = r;
#else
			v = (t << (2*Bbits)) + (l << Bbits) + r;
#endif
		}

		union {
			uint32_t v;
			struct {
				uint32_t r : Bbits;
				uint32_t l : Bbits;
				uint32_t t : 2;
			} s;
		};
	};

public:
	PVTLRBuffer()
	{
		clear();
	}

	void clear()
	{
		memset(_data, -1, length * sizeof(uint32_t));
	}

	const uint32_t &operator[](size_t i) const
	{
		return _data[i];
	}

	uint32_t &operator[](size_t i)
	{
		return _data[i];
	}

	uint32_t *get_data() { return _data; }

private:
	uint32_t _data[length];
};

}

#endif // PARALLELVIEW_PVTLRBUFFER_H
