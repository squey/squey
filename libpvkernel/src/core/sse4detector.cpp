//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/sse4detector.h>
#include <cpuid.h>
#include <cstdint>

#define SSE4_1_FLAG 0x080000
#define SSE4_2_FLAG 0x100000

static void get_cpuid_infos(CPUIDinfo* infos, int InfoType)
{
	uint32_t* eax = &infos->EAX;
	uint32_t* ebx = &infos->EBX;
	uint32_t* ecx = &infos->ECX;
	uint32_t* edx = &infos->EDX;
	__get_cpuid(InfoType, eax, ebx, ecx, edx);
}

static bool _isSSE4Supported(const unsigned int CHECKBITS)
{
	// returns true if is a Nehalem or later processor, false if prior to Nehalem

	CPUIDinfo Info{0, 0, 0, 0};
	// The code first determines if the processor is an Intel Processor.  If it is, then
	// feature flags bit 19 (SSE 4.1) and 20 (SSE 4.2) in ECX after CPUID call with EAX = 0x1
	// are checked.
	// If both bits are 1 (indicating both SSE 4.1 and SSE 4.2 exist) then
	// the function returns 1

	if (isGenuineIntel()) {
		// execute CPUID with eax (leaf) = 1 to get feature bits,
		// subleaf doesn't matter so set it to zero
		get_cpuid_infos(&Info, 0x1);
		if ((Info.ECX & CHECKBITS) == CHECKBITS) {
			return true;
		}
	}
	return false;
}

bool isSSE41Supported()
{
	const unsigned int CHECKBITS = SSE4_1_FLAG;
	return _isSSE4Supported(CHECKBITS);
}

bool isSSE41andSSE42supported()
{
	const unsigned int CHECKBITS = SSE4_1_FLAG | SSE4_2_FLAG;
	return _isSSE4Supported(CHECKBITS);
}

bool isGenuineIntel()
{
	// returns largest function # supported by CPUID if it is a Geniune Intel processor AND it
	// supports
	// the CPUID instruction, 0 if not
	CPUIDinfo Info{};
	char procString[] = "GenuineIntel";
	auto* psint = (unsigned int*)procString;
	get_cpuid_infos(&Info, 0x0);

	// execute CPUID with eax = 0, subleaf doesn't matter so set it to zero
	if ((Info.EBX == *psint) && (Info.EDX == *(psint + 1)) && (Info.ECX == *(psint + 2))) {
		return Info.EAX;
	}
	return false;
}
