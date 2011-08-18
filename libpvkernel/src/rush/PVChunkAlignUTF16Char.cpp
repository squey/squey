#include <pvkernel/rush/PVChunkAlignUTF16Char.h>
#include <unicode/ustring.h>

#include <pvkernel/core/picviz_intrin.h>

// Depends on whether SSE4.2 is enabled or not, select
// the good version of u_memechr
#ifdef __SSE4_1__
#define u_memchr_nosurr _sse_u_memchr_nosurr
#else
#define u_memchr_nosurr _u_memchr_nosurr
#endif

UChar* _u_memchr_nosurr(UChar* s, UChar c, int32_t count)
{
	if (count <= 0) {
		return NULL;
	}
	const UChar *limit=s+count;
	do { 
		if(*s==c) {
			return (UChar *)s; 
		}    
	} while(++s!=limit);
	return NULL;
} 

#ifdef __SSE4_1__
UChar* _sse_u_memchr_nosurr(UChar* s, UChar c, int32_t count)
{
	if (!has_sse41()) {
		return _u_memchr_nosurr(s, c, count);
	}
	// AG: this is an SSE optimised version of u_memchr, that
	// assume that 'c' isn't part of a surrogate pair !
	// GCC doesn't support control flow in loops when dealing
	// with vectorization. This is the output of GCC on the
	// original version (w/ -O3) :
	// 'note: not vectorized: control flow in loop.'
	
	// Number of packets of 8 UTF-16 characters
	int32_t nsse = count>>3; // >>3 <=> /8
	if (nsse <= 0) {
		return _u_memchr_nosurr(s, c, count);
	}

	__m128i sse_str,sse_cmp;
	const __m128i sse_c = _mm_set1_epi16(c);
	const __m128i sse_ff = _mm_set1_epi32(0xffffffff);
	uint64_t final_cmp[2];
	for (int32_t i = 0; i < nsse; i++) {
		// Load 8 UTF16 chars in an SSE register, and
		// search for 'c'
		sse_str = _mm_loadu_si128((__m128i*)s);
		sse_cmp = _mm_cmpeq_epi16(sse_str, sse_c);
		if (_mm_testz_si128(sse_cmp, sse_ff)) {
			s += 8;
			continue;
		}
		_mm_store_si128((__m128i*)final_cmp, sse_cmp);
		// Pseudo binary-tree based search of 0xFFFF in sse_cmp
		// "Pseudo" because no if statements is used.

		// If final_cmp[0]==0, it means that we need to search
		// in 2nd 64-bit part of our register, and that the final
		// index starts at 4.
		uint8_t reg_pos = (final_cmp[0] == 0);
		// Same with the last 2x32-bits and then 2x16-bits
		uint32_t* preg_pos_last4 = (uint32_t*) &final_cmp[reg_pos];
		uint8_t reg_pos_last4 = (preg_pos_last4[0] == 0);
		uint16_t* preg_pos_last2 = (uint16_t*) &preg_pos_last4[reg_pos_last4];
		uint8_t final_reg_pos = (reg_pos<<2) + (reg_pos_last4<<1) + (preg_pos_last2[0] == 0);
		return s+final_reg_pos;
	}
	return _u_memchr_nosurr(s, c, count-nsse*8);
}
#endif

PVRush::PVChunkAlignUTF16Char::PVChunkAlignUTF16Char(QChar c) :
	_c(c),
	_cu(c.unicode())
{
}


bool PVRush::PVChunkAlignUTF16Char::operator()(PVCore::PVChunk &cur_chunk, PVCore::PVChunk &next_chunk)
{
	//QString str_tmp = QString::fromRawData((QChar*) cur_chunk.begin(), cur_chunk.size()/sizeof(QChar));
	// Special case when _c is at the beggining of the chunk
	UChar* str_start = (UChar*) cur_chunk.begin();
	UChar* str_cur = str_start;
	ssize_t sstr = cur_chunk.size()/sizeof(UChar);
	ssize_t sstr_remains = sstr;

	if (*str_start == _c) {
		str_start++;
		sstr_remains--;
		if (sstr_remains <= 0) {
			return false;
		}
	}
	UChar* str_found;
	unsigned int nelts = 0;
	PVCore::list_elts &elts = cur_chunk.elements();
	while ((str_found = u_memchr_nosurr(str_cur, _cu, sstr_remains)) != NULL) {
		UChar* start = str_cur;
		UChar* end = str_found;
		
		cur_chunk.add_element((char*) start, (char*) end);
		nelts++;

		str_cur = str_found+1;
		sstr_remains -= ((uintptr_t)end - (uintptr_t)start)/sizeof(UChar) + 1;
	}
	if (nelts == 0) {
		PVLOG_WARN("PVChunkAlignUTF16Char: character hasn't been found !\n");
		return false;
	}
	cur_chunk.set_end((char*) (str_cur-1));

	if (sstr_remains <= 0) {
		return true;
	}
	// What's remaining goes to the next_chunk	
	memcpy(next_chunk.begin(), (char*) str_cur, sizeof(UChar)*(sstr_remains));
	next_chunk.set_end(next_chunk.begin() + sizeof(UChar)*(sstr_remains));


	return true;
}
