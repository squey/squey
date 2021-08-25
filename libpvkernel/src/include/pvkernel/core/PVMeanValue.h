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

#ifndef PVCORE_PVMEANVALUE_H
#define PVCORE_PVMEANVALUE_H

#include <cstddef>

namespace PVCore
{

/*! \brief This class computes "in live" a mean value of values added one by one
 *  \tparam T type of the values whoe mean is computed
 *  \tparam Tsum type of the sum of the values, T by default
 *
 * This class computes "in live" a mean value of values added one by one. It does not
 * store all the values but only their sum. It could compute the mean value each time and only
 * save this mean value and the number of values, but it would involve a loss of precision.
 * So, Tsum can be different that T because, for instance, T can be int16_t but the sum would be
 *int64_t.
 *
 * This a template class and each lib will include its versions. There is *no* need for
 */
template <typename T, typename Tsum = T>
class PVMeanValue
{
  public:
	PVMeanValue() : _cur_sum(0), _n_values(0) {}

	PVMeanValue(T const& v) : _cur_sum(v), _n_values(1) {}

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
		return (T)(_cur_sum / _n_values);
	}

  protected:
	Tsum _cur_sum;
	size_t _n_values;
};
} // namespace PVCore

#endif
