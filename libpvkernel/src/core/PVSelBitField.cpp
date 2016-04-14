/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/inendi_intrin.h>
#include <pvkernel/core/PVSelBitField.h>
#include <pvkernel/core/PVBitCount.h>
#include <pvkernel/core/PVHardwareConcurrency.h>
#include <pvkernel/rush/PVNraw.h>

/******************************************************************************
 *
 * PVCore::PVSelBitField::PVSelBitField()
 *
 *****************************************************************************/
PVCore::PVSelBitField::PVSelBitField(PVRow count):
	_table(nullptr),
	_count(count)
{
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::PVSelBitField(std::vector<PVRow>)
 *
 *****************************************************************************/
PVCore::PVSelBitField::PVSelBitField(std::vector<PVRow> const& r_table):
	_table(nullptr),
	_count(r_table.size())
{
	allocate_table();
	select_none();

	std::vector<PVRow>::const_iterator it;
	for (it = r_table.begin(); it != r_table.end(); it++) {
		set_line(*it, true);
	}
}

PVCore::PVSelBitField::PVSelBitField(PVSelBitField const& o):
	_table(nullptr),
	_count(o.count())
{
	allocate_and_copy_from(o);
}

PVCore::PVSelBitField::PVSelBitField(PVSelBitField&& o) :
	_table(o._table),
	_selection(o._selection),
	_count(o.count())
{
}

PVCore::PVSelBitField::operator pvcop_selection_t&() { return *static_cast<pvcop_selection_t*>(_selection); }
PVCore::PVSelBitField::operator const pvcop_selection_t&() const { return *static_cast<const pvcop_selection_t*>(_selection); }

void PVCore::PVSelBitField::ensure_allocated()
{
	if (!_table) {
		allocate_table();
	}
}

void PVCore::PVSelBitField::allocate_table()
{
	assert(_count > 0);
	assert(_count < INENDI_LINES_MAX);
	_selection = new pvcop::core::memarray<bool>(_count);
	_table = (pointer) _selection->data();
}

void PVCore::PVSelBitField::free_table()
{
	delete _selection; _table = nullptr;
}

void PVCore::PVSelBitField::copy_from(PVSelBitField const& o)
{
	assert(_table);
	assert(o._table);

	const size_t chunks = chunk_count();

	// FIXME : should use pvcop::core::array<bool>::copy_from
	__m128i sse_c;
	const PVRow chunks_multiple_of_two = chunks & ~1;
	for (size_t i = 0; i < chunks_multiple_of_two ; i += 2) {
		sse_c = _mm_load_si128((__m128i const*) &o._table[i]);
		_mm_store_si128((__m128i*) &_table[i], sse_c);
	}
	if (chunks % 2 != 0) {
		_table[chunks-1] = o._table[chunks-1];
	}
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::get_number_of_selected_lines_in_range
 *
 *****************************************************************************/
size_t PVCore::PVSelBitField::get_number_of_selected_lines_in_range(PVRow a, PVRow b) const
{
	if (!_table) {
		return 0;
	}
#if 0	
	PVRow line_index;
	size_t count = 0;

	for (line_index = a; line_index<b; line_index++) { 
		if (get_line(line_index)) { 
			count++;
		}
	}

	return count;
#endif
	assert(b > a);
	
	return PVCore::PVBitCount::bit_count_between(a, b-1, &_table[0]);
}

std::vector<PVRow> PVCore::PVSelBitField::get_rows_table()
{
	if (!_table) {
		return std::vector<PVRow>();
	}

	std::vector<PVRow> r_table;
	r_table.reserve(count());

	PVRow line_index;

	for (line_index = 0; line_index < count(); line_index++) {
		if (get_line(line_index)) {
			r_table.push_back(line_index);
		}
	}

	return r_table;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::is_empty
 *
 *****************************************************************************/
bool PVCore::PVSelBitField::is_empty() const
{
	if (!_table) {
		return true;
	}
	const size_t chunks = chunk_count();
#ifdef __SSE4_1__
	const __m128i ones = _mm_set1_epi32(0xFFFFFFFF);
	__m128i vec;
	for (PVRow i = 0; i < chunks; i += 2) {
		vec = _mm_load_si128((__m128i*) &_table[i]);
		if (_mm_testz_si128(vec, ones) == 0) {
			return false;
		}
	}
if (chunks % 2 != 0)
	for (PVRow i = (chunks/2)*2; i < chunks; i++) {
		if (_table[i] != 0) {
			return false;
		}
	}
#else
	for (PVRow i = 0; i < chunks; i++) {
		if (_table[i] != 0) {
			return false;
		}
	}
#endif
	return true;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::is_empty_between
 *
 *****************************************************************************/
bool PVCore::PVSelBitField::is_empty_between(PVRow const a, PVRow b) const
{
	assert(b > a);
	assert(b <= count());

	if (!_table) {
		return true;
	}
	
	// 'b' is not included
	b--;

	size_t chunk_start = a/CHUNK_SIZE;
	const size_t chunk_end   = b/CHUNK_SIZE;
	
	const PVRow cbit = line_index_to_chunk_bit(a);
	if (cbit > 0) {
		// Prelogue
		uint64_t cv = _table[chunk_start];
		if (chunk_end == chunk_start) {
			const size_t off = CHUNK_SIZE-line_index_to_chunk_bit(b)-1;
			cv = (cv << off) >> off;
			return (cv >> cbit) == 0;
		}
		if ((cv >> cbit) != 0) {
			return false;
		}
		chunk_start++;
	}

	// Main loop
	for (size_t c = chunk_start; c < chunk_end; c++) {
		if (_table[c] != 0) {
			return false;
		}
	}

	// Epilogue
	return (_table[chunk_end] << (CHUNK_SIZE-line_index_to_chunk_bit(b)-1)) == 0;
}

bool PVCore::PVSelBitField::operator==(const PVSelBitField &rhs) const
{
	if (&rhs == this) {
		return true;
	}

	if ((_table == nullptr) ^ (rhs._table == nullptr)) {
		return false;
	}
	if ((_table == nullptr) && (rhs._table == nullptr)) {
		return true;
	}

	assert(count() == rhs.count());

	return memcmp(_table, rhs._table, _selection->mem_size()) == 0;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator=
 *
 *****************************************************************************/
PVCore::PVSelBitField& PVCore::PVSelBitField::operator=(const PVSelBitField &rhs)
{
	if (&rhs == this) {
		return *this;
	}

	if (rhs._table) {
		if (!_table) {
			allocate_table();
		}
		copy_from(rhs);
	}
	else {
		if (_table) {
			memset(_table, 0x00, _selection->mem_size());
		}
	}

	return *this;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator&
 *
 *****************************************************************************/
PVCore::PVSelBitField PVCore::PVSelBitField::operator&(const PVSelBitField &rhs) const
{
	PVCore::PVSelBitField result(*this);
	result &= rhs;

	return result;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator&=
 *
 *****************************************************************************/
PVCore::PVSelBitField & PVCore::PVSelBitField::operator&=(const PVSelBitField &rhs)
{
	if (!_table) {
		allocate_table();
		select_none();
		return *this;
	}

	assert(chunk_count() == rhs.chunk_count());

	const size_t chunks = chunk_count();
	for (PVRow i = 0; i < chunks; i++) {
		_table[i] &= rhs._table[i];
	}

	return *this;
}

PVCore::PVSelBitField& PVCore::PVSelBitField::and_optimized(const PVSelBitField& rhs)
{
	if (!_table) {
		allocate_table();
		select_none();
		return *this;
	}

	assert(chunk_count() == rhs.chunk_count());

	const ssize_t last_chunk = get_max_last_nonzero_chunk_index(rhs);
	if (last_chunk >= 0) {
		for (PVRow i = 0; i < (PVRow) last_chunk; i++) {
			_table[i] &= rhs._table[i];
		}
	}

	return *this;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator~
 *
 *****************************************************************************/
PVCore::PVSelBitField PVCore::PVSelBitField::operator~() const
{
	PVSelBitField result(count());
	result.allocate_table();
	if (!_table) {
		result.select_all();
	}
	else {
		assert(chunk_count() == result.chunk_count());
		const size_t chunks = chunk_count();
		for (PVRow i = 0; i < chunks; i++) {
			result._table[i] = ~_table[i];
		}
	}

	return result;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator|
 *
 *****************************************************************************/
PVCore::PVSelBitField PVCore::PVSelBitField::operator|(const PVSelBitField &rhs) const
{
	assert(chunk_count() == rhs.chunk_count());

	PVSelBitField result(*this);
	result |= rhs;

	return result;
}

/******************************************************************************
*
 * PVCore::PVSelBitField::operator|=
 *
 *****************************************************************************/
PVCore::PVSelBitField & PVCore::PVSelBitField::operator|=(const PVSelBitField &rhs)
{ 
	if ((&rhs == this) || (rhs._table == NULL)) {
		return *this;
	}

	if (!_table) {
		allocate_table();
		copy_from(rhs);
		return *this;
	}

	assert(chunk_count() == rhs.chunk_count());

	const size_t chunks = chunk_count();
	for (PVRow i = 0; i < chunks; i++) {
		_table[i] |= rhs._table[i];
	}

	return *this;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator-
 *
 *****************************************************************************/
PVCore::PVSelBitField PVCore::PVSelBitField::operator-(const PVSelBitField &rhs) const
{
	PVSelBitField result = *this;
	result -= rhs;

	return result;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator-=
 *
 *****************************************************************************/
PVCore::PVSelBitField & PVCore::PVSelBitField::operator-=(const PVSelBitField &rhs)
{
	if ((&rhs == this) || (rhs._table == NULL)) {
		return *this;
	}

	if (!_table) {
		allocate_table();
		select_none();
		return *this;
	}

	assert(chunk_count() == rhs.chunk_count());

	const size_t chunks = chunk_count();
	for (PVRow i = 0; i < chunks; i++) {
		_table[i] &= ~rhs._table[i];
	}

	return *this;
}

void PVCore::PVSelBitField::AB_sub(PVSelBitField const& a, PVSelBitField const& b)
{
	static size_t nthreads = PVCore::PVHardwareConcurrency::get_physical_core_number();

	if (!_table) {
		allocate_table();
	}

	if (!a._table) {
		select_none();
		return;
	}

	if (!b._table) {
		select_all();
		return;
	}

	assert(chunk_count() == a.chunk_count() && chunk_count() == b.chunk_count());

	const size_t chunks = chunk_count();

#pragma omp parallel for num_threads(nthreads)
	for (PVRow i = 0; i < chunks; i++) {
		_table[i] = a._table[i] & (~b._table[i]);
	}
}

size_t PVCore::PVSelBitField::chunk_count() const
{
	return _selection->mem_size() / (CHUNK_SIZE / CHUNK_SIZE_BYTE);
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator^
 *
 *****************************************************************************/
PVCore::PVSelBitField PVCore::PVSelBitField::operator^(const PVSelBitField &rhs) const
{
	PVSelBitField result = *this;
	result ^= rhs;

	return result;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator^=
 *
 *****************************************************************************/
PVCore::PVSelBitField & PVCore::PVSelBitField::operator^=(const PVSelBitField &rhs)
{
	if ((&rhs == this) || (rhs._table == NULL)) {
		return *this;
	}

	if (!_table) {
		allocate_table();
		copy_from(rhs);
		return *this;
	}

	const size_t chunks = chunk_count();
	for (PVRow i = 0; i < chunks; i++) {
		_table[i] ^= rhs._table[i];
	}

	return *this;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::or_not
 *
 *****************************************************************************/
PVCore::PVSelBitField & PVCore::PVSelBitField::or_not(const PVSelBitField &rhs)
{
	if ((&rhs == this)) {
		return *this;
	}

	if (!rhs._table) {
		select_all();
		return *this;
	}

	const size_t chunks = chunk_count();

	if (!_table) {
		allocate_table();
		for (PVRow i = 0; i < chunks; i++) {
			_table[i] = ~(rhs._table[i]);
		}
		return *this;
	}

	for (PVRow i = 0; i < chunks; i++) {
		_table[i] |= ~(rhs._table[i]);
	}

	return *this;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::and_not
 *
 *****************************************************************************/
PVCore::PVSelBitField & PVCore::PVSelBitField::and_not(const PVSelBitField &rhs)
{
	if ((&rhs == this) || (rhs._table == NULL)) {
		return *this;
	}

	if (!_table) {
		allocate_table();
		select_none();
		return *this;
	}

	const size_t chunks = chunk_count();
	for (PVRow i = 0; i < chunks; i++) {
		_table[i] &= ~(rhs._table[i]);
	}

	return *this;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::xor_not
 *
 *****************************************************************************/
PVCore::PVSelBitField & PVCore::PVSelBitField::xor_not(const PVSelBitField &rhs)
{
	if (&rhs == this) {
		return *this;
	}

	if (!rhs._table) {
		select_all();
		return *this;
	}

	const size_t chunks = chunk_count();

	if (!_table) {
		allocate_table();
		for (PVRow i = 0; i < chunks; i++) {
			_table[i] = ~(rhs._table[i]);
		}
		return *this;
	}

	for (PVRow i = 0; i < chunks; i++) {
		_table[i] ^= ~(rhs._table[i]);
	}

	return *this;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::select_all
 *
 *****************************************************************************/
void PVCore::PVSelBitField::select_all()
{
	if (!_table) {
		allocate_table();
	}
	memset(_table, 0xFF, _selection->mem_size());
}

void PVCore::PVSelBitField::select_random()
{
	if (!_table) {
		allocate_table();
	}
	const size_t chunks = chunk_count();
	for (PVRow i = 0; i < chunks; i++) {
		_table[i] = (uint64_t) (rand()) | ((uint64_t)(rand()) << 32);
	}
}

void PVCore::PVSelBitField::select_random(const PVRow n)
{
	if (!_table) {
		allocate_table();
	}

	select_none();

	for (PVRow i = 0; i < n; i++) {
		set_bit_fast(rand() % count());
	}
}



/******************************************************************************
 *
 * PVCore::PVSelBitField::select_even
 *
 *****************************************************************************/
void PVCore::PVSelBitField::select_even()
{
	if (!_table) {
		allocate_table();
	}
	memset(_table, 0xAA, _selection->mem_size());
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::select_none
 *
 *****************************************************************************/
void PVCore::PVSelBitField::select_none()
{
	if (!_table) {
		allocate_table();
	}
	memset(_table, 0x00, _selection->mem_size());
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::select_odd
 *
 *****************************************************************************/
void PVCore::PVSelBitField::select_odd()
{
	if (!_table) {
		allocate_table();
	}
	memset(_table, 0x55, _selection->mem_size());
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::select_byte_pattern
 *
 *****************************************************************************/
void PVCore::PVSelBitField::select_byte_pattern(const unsigned char byte_pattern)
{
	if (!_table) {
		allocate_table();
	}
	memset(_table, byte_pattern, _selection->mem_size());
}
/******************************************************************************
 *
 * PVCore::PVSelBitField::select_inverse
 *
 *****************************************************************************/
void PVCore::PVSelBitField::select_inverse()
{
	if (!_table) {
		allocate_table();
		select_all();
		return;
	}

	const size_t chunks = chunk_count();
	for (PVRow i = 0; i < chunks; i++) {
		_table[i] = ~_table[i];
	}
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::set_line
 *
 *****************************************************************************/
void PVCore::PVSelBitField::set_line(PVRow line_index, bool bool_value)
{
	if (!_table) {
		allocate_table();
		select_none();
	}

	const PVRow pos = line_index_to_chunk(line_index);
	const PVRow shift = line_index_to_chunk_bit(line_index);
	
	if ( bool_value )  {
		_table[pos] |= 1UL<<shift;
	} else {
		_table[pos] &= ~(1UL<<shift);
	}
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::set_line_select_only
 *
 *****************************************************************************/
void PVCore::PVSelBitField::set_line_select_only(PVRow line_index, bool bool_value)
{
	if (bool_value == false) {
		return;
	}

	set_line(line_index, true);
}

ssize_t PVCore::PVSelBitField::get_last_nonzero_chunk_index(ssize_t starting_chunk, ssize_t ending_chunk) const
{
	if (ending_chunk == -1) {
		ending_chunk = chunk_count() -1 ;
	}

	if (!_table || (starting_chunk < 0 || ending_chunk < 0)) {
		return -1;
	}
#ifdef __SSE4_1__
	const __m128i ones = _mm_set1_epi32(0xFFFFFFFF);
	__m128i vec;
	const ssize_t ending_chunk_aligned = (ssize_t)(((size_t)ending_chunk>>1)<<1);
	if (ending_chunk_aligned <= starting_chunk) {
		for (ssize_t i = ending_chunk; i >= starting_chunk; i--) {
			if (_table[i] != 0) {
				return i;
			}
		}
	}
	else {
		for (ssize_t i = ending_chunk; i >= ending_chunk_aligned; i--) {
			if (_table[i] != 0) {
				return i;
			}
		}
		const ssize_t starting_chunk_aligned = ((starting_chunk+3)/2)*2;
		for (ssize_t i = ((ssize_t)ending_chunk_aligned)-2; i >= starting_chunk_aligned; i -= 2) {
			vec = _mm_load_si128((__m128i*) &_table[i]);
			if (_mm_testz_si128(vec, ones) == 0) {
				uint64_t DECLARE_ALIGN(16) final_sel[2];
				_mm_store_si128((__m128i*)final_sel, vec);
				// If final_sel[0] == 0, it means that the chunk is in one of the last two 32 bits of vec.
				// Otherwise, it is one of the two first.
				uint8_t reg_pos = (final_sel[1] != 0);
				//uint32_t* preg_last2 = (uint32_t*) &final_sel[reg_pos];
				//uint8_t reg_pos_last2 = (preg_last2[1] != 0);
				//return i + (reg_pos<<1) + reg_pos_last2;
				return i + (reg_pos<<1);
			}
		}
		for (ssize_t i = starting_chunk_aligned-1; i >= starting_chunk; i--) {
			if (_table[i] != 0) {
				return i;
			}
		}
	}
	return starting_chunk-1;

#else
	for (ssize_t i = ending_chunk; i >= starting_chunk; i--) {
		if (_table[i] != 0) {
			return i;
		}
	}
	return starting_chunk-1;
#endif
}

PVCore::PVSelBitField& PVCore::PVSelBitField::or_optimized(const PVSelBitField& rhs)
{
	if ((&rhs == this) || (!rhs._table)) {
		return *this;
	}

	if (!_table) {
		allocate_table();
		copy_from(rhs);
		return *this;
	}

	assert(count() == rhs.count());

	const ssize_t last_chunk = rhs.get_last_nonzero_chunk_index();
	if (last_chunk >= 0) {
		// TODO: GCC vectorize this, but we could try to
		// do this by SSE and checking whether rhs._table[] vectors
		// are non-null before doing the actual OR+store (thus saving stores)
		for (PVRow i = 0; i <= (PVRow) last_chunk; i++) {
			_table[i] |= rhs._table[i];
		}
	}

	return *this;
}

PVCore::PVSelBitField& PVCore::PVSelBitField::or_range(const PVSelBitField &rhs,
                                                       PVRow start, PVRow end)
{
	if ((&rhs == this) || (!rhs._table)) {
		return *this;
	}

	if (!_table) {
		allocate_table();
		copy_from(rhs);
		return *this;
	}

	const PVRow chunk_start = line_index_to_chunk(start);
	const PVRow chunk_end = line_index_to_chunk(end);

	for (PVRow i = chunk_start; i <= chunk_end; ++i) {
		_table[i] |= rhs._table[i];
	}

	return *this;
}


ssize_t PVCore::PVSelBitField::get_min_last_nonzero_chunk_index(PVSelBitField const& other) const
{
	if (!_table) {
		return -1;
	}
	const ssize_t last_chunk = other.get_last_nonzero_chunk_index();
	return get_last_nonzero_chunk_index(0, last_chunk);
}

ssize_t PVCore::PVSelBitField::get_max_last_nonzero_chunk_index(PVSelBitField const& other) const
{
	if (!_table) {
		return -1;
	}
	const ssize_t last_chunk = other.get_last_nonzero_chunk_index();
	return get_last_nonzero_chunk_index(last_chunk, chunk_count()-1);
}


PVRow PVCore::PVSelBitField::find_next_set_bit(const PVRow index, const PVRow size) const
{
	/**
	 * This function can be optimized by testing chunk per chunk and using __builtin_ctzl
	 * but it takes really few ms to compute on 200 Me
	 */
	if (index > size) {
		return PVROW_INVALID_VALUE;
	}

	for(PVRow i = index; i < size; ++i) {
		if (get_line_fast(i))
			return i;
	}

	return PVROW_INVALID_VALUE;
}

PVRow PVCore::PVSelBitField::find_previous_set_bit(const PVRow index, const PVRow size) const
{
	/**
	 * This function can be optimized by testing chunk per chunk and using __builtin_clzl
	 * but it takes really few ms to compute on 200 Me
	 */
	if (index > size) {
		return PVROW_INVALID_VALUE;
	}

	for(PVRow i = index; i > 0; --i) {
		if (get_line_fast(i))
			return i;
	}

	// as unsigned can not be negative, the case 0 is tested outside of the loop
	if (get_line_fast(0))
		return 0;

	return PVROW_INVALID_VALUE;
}

void PVCore::PVSelBitField::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.attribute("count", _count);
	size_t mem_size = pvcop::core::__impl::bit::to_mem_size(_count);
	if (so.is_writing()) {
		if (_table) {
			so.buffer("selection_data", _table, mem_size);
		}
	}
	else {
		if (so.buffer_exists("selection_data")) {
			if (!_table) {
				allocate_table();
			}
			so.buffer("selection_data", _table, mem_size);
		}
		else {
			if (_table) {
				select_none();
			}
		}
	}
}
