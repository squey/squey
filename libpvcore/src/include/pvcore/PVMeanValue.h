#ifndef PVCORE_PVMEANVALUE_H
#define PVCORE_PVMEANVALUE_H

#include <pvcore/general.h>

namespace PVCore {

/*! \brief This class computes "in live" a mean value of values added one by one
 *  \tparam T type of the values whoe mean is computed
 *  \tparam Tsum type of the sum of the values, T by default
 *
 * This class computes "in live" a mean value of values added one by one. It does not
 * store all the values but only their sum. It could compute the mean value each time and only
 * save this mean value and the number of values, but it would involve a loss of precision.
 * So, Tsum can be different that T because, for instance, T can be int16_t but the sum would be int64_t.
 *
 * This a template class and each lib will export its version. There is *no* need for LibCoreDecl
 */
template<typename T, typename Tsum = T>
class PVMeanValue
{
public:
	PVMeanValue():
		_cur_sum(0),
		_n_values(0)
	{
	}

	PVMeanValue(T const& v):
		_cur_sum(v),
		_n_values(1)
	{
	}

	inline void push(T const& v)
	{
		_cur_sum += v;
		_n_values++;
	}

	inline T compute_mean() const
	{
		if (_n_values == 0) {
			return 0;
		}
		return (T) (_cur_sum/_n_values);
	}

protected:
	Tsum _cur_sum;
	size_t _n_values;
};

}

#endif
