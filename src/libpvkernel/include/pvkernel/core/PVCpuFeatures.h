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

#ifndef PVKERNEL_CORE_PVCPUFEATURES_H
#define PVKERNEL_CORE_PVCPUFEATURES_H

namespace PVCore
{

/**
 * Runtime instruction set detection, for kernels compiled twice.
 *
 * The binaries target x86-64-v2 so that they still start on the pre-Haswell machines
 * the crash reports come from. A hot kernel can nonetheless be built a second time
 * with -mavx2 in its own translation unit and selected here, which keeps the wider
 * instructions off the machines that lack them without holding back the others.
 *
 * Setting SQUEY_FORCE_BASELINE in the environment pins the baseline path. The test
 * suite uses it to check both paths agree, and it doubles as an escape hatch should a
 * vectorised path ever misbehave in the field.
 */
bool has_avx2();

} // namespace PVCore

#endif // PVKERNEL_CORE_PVCPUFEATURES_H
