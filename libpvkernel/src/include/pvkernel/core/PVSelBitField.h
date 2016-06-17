/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVBITFIELD_H
#define PVCORE_PVBITFIELD_H

#include <pvkernel/core/inendi_intrin.h>
#include <pvkernel/core/PVAllocators.h>
#include <pvkernel/core/PVAlgorithms.h>
#include <pvkernel/core/PVBitVisitor.h>
#include <pvkernel/core/PVSerializeArchive.h>

#include <pvcop/core/memarray.h>

namespace pvcop
{
namespace core
{

template <typename T>
class array;
}
}

#include <vector>

namespace Inendi
{
class PVSelection;
}

namespace PVCore
{

/**
* \class PVSelBitField
*/

class PVSelBitField
{
	friend class PVCore::PVSerializeObject;
	friend class Inendi::PVSelection;

  public:
	static constexpr auto CHUNK_SIZE = pvcop::core::__impl::bit::chunk_bit_size;
	static constexpr auto CHUNK_SIZE_BYTE = pvcop::core::__impl::bit::chunk_byte_size;

  public:
	typedef uint64_t chunk_t;
	typedef chunk_t DECLARE_ALIGN(16) * pointer;
	typedef chunk_t DECLARE_ALIGN(16) const* const_pointer;
	typedef PVCore::PVAlignedAllocator<chunk_t, 16> allocator;
	typedef pvcop::core::array<bool> pvcop_selection_t;

  public:
	/**
	 * Constructor
	 */
	explicit PVSelBitField(PVRow count);

	/**
	 * Create a PVSelBitField object from a row table until we reach '0' marking
	 * the end of this row table.
	 *
	 * The PVSelBitField is a binary '0100' if the second line is selected.
	 */
	explicit PVSelBitField(std::vector<PVRow> const& rtable);

	PVSelBitField(PVSelBitField const& o);

	~PVSelBitField()
	{
		if (_table) {
			free_table();
		}
	}

	// Move constructor. Save a lot of useless allocations, memcpys and desallocations !
	PVSelBitField(PVSelBitField&& o);

	operator pvcop_selection_t&();
	operator const pvcop_selection_t&() const;

	void set_count(PVRow count) { _count = count; }
	PVRow count() const { return _count; }

	size_t chunk_count() const;

	/*! \brief Ensure that selection buffer is allocated.
	 *
	 * This function will allocate the selection buffer if it has not already been.
	 *
	 * Warning: in the case the allocation is performed, the content of the
	 * selection is undetermined! It is the responsability of the caller to
	 * call next select_none, select_all or do anything he likes with the
	 * selection. This is done for performance reason, so that in some
	 * situations, the selection buffer is only written once (and a useless
	 * select_none is avoided).
	 */
	void ensure_allocated();

	/**
	 * Destructor
	 */

	inline pointer get_buffer() { return _table; }
	inline const_pointer get_buffer() const { return _table; };

	/**
	 * Get the state of line N in the PVSelBitField
	 *
	 * @param line_index The index of the line we are interested in
	 *
	 * @return A boolean stating whether it is set or not
	 */
	inline bool get_line(PVRow line_index) const
	{
		if (!_table) {
			return false;
		}
		const PVRow pos = line_index_to_chunk(line_index);
		const PVRow shift = line_index_to_chunk_bit(line_index);

		/*
		 * Say you want to retrieve if the line 20000 is selected or not:
		 * pos = 312
		 * shift = 32
		 */

		return (_table[pos] & (1UL << shift));
	}

	/**
	 * Fast version of get_line() (i.e. no check for _table existence)
	 */
	inline bool get_line_fast(PVRow line_index) const
	{
		const PVRow pos = line_index_to_chunk(line_index);
		const PVRow shift = line_index_to_chunk_bit(line_index);
		return (_table[pos] & (1UL << shift));
	}

	/**
	 * Get the state of C lines starting at line N in the PVSelBitField
	 *
	 * This method must be used with caution:
	 * - bits can not be extracted accross chunks
	 * - C must be a power of two
	 * - L must be a multiple of C
	 *
	 * @param line_index The index of the first line we are interested in
	 * @param count The number of lines we are interested in
	 *
	 * @return An unsigned stating whether they are set or not
	 */
	inline uint32_t get_lines_fast(PVRow line_index, uint32_t count) const
	{
		assert(PVCore::is_power_of_two(count));
		assert((line_index % count) == 0);

		const PVRow pos = line_index_to_chunk(line_index);
		const PVRow shift = line_index_to_chunk_bit(line_index);

		return (_table[pos] >> shift) & ((1UL << count) - 1);
	}

	/**
	 * Gets the number of lines that are selected, in the range [a,b[
	 * So b is NOT INCLUDED !
	 *
	 * @param a The index of the first line in the range
	 * @param b The index of the line after the last one in the range (NOT INCLUDED !)
	 *
	 * @return The number of selected line in that range
	 */
	size_t get_number_of_selected_lines_in_range(PVRow a, PVRow b) const;

	bool is_empty() const;
	// Check whether a selection is empty between bits 'a' and 'b'. 'b' is *not* included.
	bool is_empty_between(PVRow const a, PVRow const b) const;

	/**
	 * Select random lines
	 */
	void select_random();

	/**
	 * Select randomly `n' lines
	 */
	void select_random(const PVRow n);

	bool operator==(const PVSelBitField& rhs) const;

	/**
	 * This is the copy operator
	 * B = A
	 *
	 * @param &rhs
	 *
	 * @return The resulting PVSelBitField
	 */
	PVSelBitField& operator=(const PVSelBitField& rhs);

	PVSelBitField& operator=(PVSelBitField&& rhs)
	{
		if (this != &rhs) {
			if (_table) {
				free_table();
			}
			_table = rhs._table;
			rhs._table = nullptr;

			_selection = rhs._selection;
			rhs._selection = nullptr;
		}

		return *this;
	}

	/**
	 * This is the binary outplaced 'AND' operation on two selections
	 * C = A & B
	 *
	 * @param rhs The second selection involved in the 'AND' operator
	 *
	 * @return The resulting PVSelBitField
	 */
	PVSelBitField operator&(const PVSelBitField& rhs) const;

	/**
	 * This is the binary INPLACED 'AND' operation on two selections
	 * A &= B  (that stands for A = A & B)
	 *
	 * @param rhs The second selection involved in the 'AND' operator
	 *
	 * @return A reference to the resulting PVSelBitField
	 */
	PVSelBitField& operator&=(const PVSelBitField& rhs);
	PVSelBitField& and_optimized(const PVSelBitField& rhs);

	/**
	 * This is the unary bitwise outplaced 'NOT' operation on one selection
	 * B = ~A
	 *
	 * @return The resulting PVSelBitField
	 */
	PVSelBitField operator~() const;

	/**
	 * This is the binary outplaced 'OR' operation on two selections
	 * C = A | B
	 *
	 * @param rhs The second selection involved in the 'OR' operator
	 *
	 * @return The resulting PVSelBitField
	 */
	PVSelBitField operator|(const PVSelBitField& rhs) const;

	/**
	 * This is the binary INPLACED 'OR' operation on two selections
	 * A |= B  (that stands for A = A | B)
	 *
	 * @param rhs The second selection involved in the 'OR' operator
	 *
	 * @return A reference to the resulting PVSelBitField
	 */
	PVSelBitField& operator|=(const PVSelBitField& rhs);
	PVSelBitField& or_optimized(const PVSelBitField& rhs);
	PVSelBitField& or_range(const PVSelBitField& rhs, PVRow start, PVRow end);

	/**
	 * This is the binary outplaced 'SUBSTRACTION' operation on two selections
	 * C = A - B
	 *
	 * @param rhs The second selection involved in the 'SUBSTRACTION' operator
	 *
	 * @return The resulting PVSelBitField
	 */
	PVSelBitField operator-(const PVSelBitField& rhs) const;

	/**
	 * This is the binary INPLACED 'SUBSTRACTION' operation on two selections
	 * A -= B  (that stands for A = A - B)
	 *
	 * @param rhs The second selection involved in the 'SUBSTRACTION' operator
	 *
	 * @return A reference to the resulting PVSelBitField
	 */
	PVSelBitField& operator-=(const PVSelBitField& rhs);

	void AB_sub(PVSelBitField const& a, PVSelBitField const& b);

	/**
	 * This is the binary outplaced 'XOR' operation on two selections
	 * C = A ^ B
	 *
	 * @param rhs The second selection involved in the 'XOR' operator
	 *
	 * @return The resulting PVSelBitField
	 */
	PVSelBitField operator^(const PVSelBitField& rhs) const;

	/**
	 * This is the binary INPLACED 'XOR' operation on two selections
	 * A ^= B  (that stands for A = A ^ B)
	 *
	 * @param rhs The second selection involved in the 'XOR' operator
	 *
	 * @return A reference to the resulting PVSelBitField
	 */
	PVSelBitField& operator^=(const PVSelBitField& rhs);

	/**
	 * This is the binary INPLACED 'OR NOT' operation on two selections
	 * A.or_not(B)  (that stands for A = A | ~B)
	 *
	 * @param rhs The second selection involved in the 'OR NOT' operator
	 *
	 * @return A reference to the resulting PVSelBitField
	 */
	PVSelBitField& or_not(const PVSelBitField& rhs);

	/**
	 * This is the binary INPLACED 'AND NOT' operation on two selections
	 * A.and_not(B)  (that stands for A = A | ~B)
	 *
	 * @param rhs The second selection involved in the 'AND NOT' operator
	 *
	 * @return A reference to the resulting PVSelBitField
	 */
	PVSelBitField& and_not(const PVSelBitField& rhs);

	/**
	 * This is the binary INPLACED 'XOR NOT' operation on two selections
	 * A.xor_not(B)  (that stands for A = A ^ ~B)
	 *
	 * @param rhs The second selection involved in the 'XOR NOT' operator
	 *
	 * @return A reference to the resulting PVSelBitField
	 */
	PVSelBitField& xor_not(const PVSelBitField& rhs);

	void select_all();
	void select_even();
	void select_none();
	void select_odd();
	void select_inverse();
	void select_byte_pattern(const unsigned char byte_pattern);

	/**
	 * Sets the state of line N in the PVSelBitField
	 *
	 * @param line_index The index of the line we are interested in
	 * @param bool_value The boolean value we want to set for that line_index
	 *
	 */
	void set_line(PVRow line_index, bool bool_value);

	/**
	 * Sets the state of line N in the PVSelBitField only if bool_value is true
	 *
	 * @param line_index The index of the line we are interested in
	 * @param bool_value The boolean value we want to set for that line_index
	 *
	 */
	void set_line_select_only(PVRow line_index, bool bool_value);

	inline void set_bit_fast(PVRow const line_index)
	{
		_table[line_index_to_chunk(line_index)] |= 1UL << (line_index_to_chunk_bit(line_index));
	}
	inline void clear_bit_fast(PVRow const line_index)
	{
		_table[line_index_to_chunk(line_index)] &= ~(1UL << (line_index_to_chunk_bit(line_index)));
	}

	/*! \brief Set the four bits of \a bits from line_index
	 *
	 * Warning: this can't be used to set bits accross chunk !
	 */
	inline void set_4bit_fast(PVRow const line_index, const uint32_t bits)
	{
		const PVRow pos = line_index_to_chunk(line_index);
		const PVRow shift = line_index_to_chunk_bit(line_index);
		assert(shift <= (CHUNK_SIZE - 4));

		_table[pos] |= ((chunk_t)bits << shift);
	}

	/*! \brief Get a complete 32-bit chunk
	 *  \param chunk_index The index of the chunk as if chunks were 32-bits wide !
	 *
	 * Get a complete 32-bit chunk.
	 *
	 * \warning The chunk_index given here must be computed as if chunks were
	 * 32 bits wide ! The line_index_to_chunk32 function can help with that.
	 */
	inline uint32_t get_chunk32_fast(PVRow const chunk_index) const
	{
		return reinterpret_cast<uint32_t const*>(_table)[chunk_index];
	}

	/*! \brief Get a complete bit chunk
	 *  \param chunk_index The index of the chunk
	 *
	 * Get a complete chunk.
	 */
	inline uint64_t get_chunk_fast(PVRow const chunk_index) const
	{
		assert(chunk_index < chunk_count());
		return _table[chunk_index];
	}

	/*! \brief Set a complete chunk
	 *  \param chunk_index The index of the chunk (warning, not the line index !)
	 *
	 * Set a complete chunk.
	 */
	inline void set_chunk_fast(PVRow const chunk_index, chunk_t const chunk)
	{
		assert(chunk_index < chunk_count());
		_table[chunk_index] = chunk;
	}

	/*! \brief Set a complete chunk using one stream instructions
	 *  \param chunk_index The index of the chunk as if chunks were 32-bits wide !
	 *
	 * Set a complete chunk.
	 * \warning The chunk_index given here must be computed as if chunks were
	 * 32 bits wide ! The line_index_to_chunk32 function can help with that.
	 */
	inline void set_chunk32_fast_stream(PVRow const chunk_index, int32_t const chunk)
	{
		_mm_stream_si32(&(reinterpret_cast<int32_t*>(_table)[chunk_index]), chunk);
	}

	// Returns the index of the chunk following the last chunk that contains a line
	// Thus, returns 0 if no chunk is empty
	ssize_t get_last_nonzero_chunk_index(ssize_t starting_chunk = 0,
	                                     ssize_t ending_chunk = -1) const;

	/**
	 * search forward for the first bit set to 1 from the position \a index.
	 *
	 * the seach is also done in [index;size[
	 *
	 * @param index the search's start position
	 * @param size the selection's size
	 *
	 * @return the index of the first next bit set to 1; PVROW_INVALID_VALUE otherwise.
	 */
	PVRow find_next_set_bit(const PVRow index, const PVRow size) const;

	/**
	 * search backward for the first bit set to 1 from the position \a index.
	 *
	 * the seach is also done in [0;index] in the reverse order
	 *
	 * @param index the search's start position
	 * @param size the selection's size
	 *
	 * @return the index of the first previous bit set to 1; PVROW_INVALID_VALUE otherwise.
	 */
	PVRow find_previous_set_bit(const PVRow index, const PVRow size) const;

	template <class F>
	void visit_selected_lines(F const& f, PVRow b = PVROW_INVALID_VALUE, const PVRow a = 0) const
	{
		if (b == PVROW_INVALID_VALUE) {
			b = count();
		}

		if (!_table || (b <= 0)) {
			return;
		}
		assert(b > a);
		assert(b <= count());
		b--;
		PVRow last_bit = line_index_to_chunk_bit(b);
		const ssize_t org_chunk_end = line_index_to_chunk(b);
		ssize_t chunk_start = line_index_to_chunk(a);
		ssize_t chunk_end = get_last_nonzero_chunk_index(chunk_start, org_chunk_end);
		if (chunk_end < chunk_start) {
			// No lines are selected !
			return;
		}
		if (chunk_end != org_chunk_end) {
			last_bit = CHUNK_SIZE - 1;
		}

		// If there are less than or exactly 3 chunks, use the serial version
		if ((chunk_end - chunk_start + 1) <= 3) {
			PVRow new_b = b + 1;
			if (line_index_to_chunk(b) > chunk_end) {
				new_b = chunk_to_line_index(chunk_end + 1);
			}
			visit_selected_lines_serial(f, new_b, a);
			return;
		}

		const PVRow cbit = line_index_to_chunk_bit(a);
		if (cbit > 0) {
			// Prelogue
			uint64_t cv = _table[chunk_start];
			PVCore::PVBitVisitor::visit_bits((cv >> cbit) << cbit, f,
			                                 chunk_to_line_index(chunk_start));
			chunk_start++;
		}

		// Main loop
		ssize_t c;
		const ssize_t chunk_start_aligned = (chunk_start + 1) & (~(ssize_t)1);
		const ssize_t chunk_end_sse = chunk_end & (~(ssize_t)1);
		for (c = chunk_start; c < chunk_start_aligned; c++) {
			const uint64_t sel_buf = _table[c];
			PVCore::PVBitVisitor::visit_bits(sel_buf, f, chunk_to_line_index(c));
		}
		for (; c < chunk_end_sse; c += 2) {
			const __m128i sse_sel = _mm_load_si128((__m128i*)&_table[c]);
			PVCore::PVBitVisitor::visit_bits(sse_sel, f, chunk_to_line_index(c));
		}
		for (; c < chunk_end; c++) {
			const uint64_t sel_buf = _table[c];
			PVCore::PVBitVisitor::visit_bits(sel_buf, f, chunk_to_line_index(c));
		}

		// Epilogue
		uint64_t last_chunk = _table[chunk_end];
		if (last_bit < CHUNK_SIZE - 1) {
			last_chunk &= ((1ULL << (last_bit + 1)) - 1);
		}
		PVCore::PVBitVisitor::visit_bits(last_chunk, f, chunk_to_line_index(chunk_end));
	}

	/**
	 * Get the float table from PVSelBitField.
	 */
	std::vector<PVRow> get_rows_table();

	static inline PVRow line_index_to_chunk(const PVRow r) { return r >> 6; }
	static inline PVRow line_index_to_chunk32(const PVRow r) { return r >> 5; }
	static inline PVRow chunk_to_line_index(const PVRow r) { return r << 6; }
	static inline PVRow line_index_to_chunk_bit(const PVRow r) { return r & 63; }

  public:
	template <class F>
	void
	visit_selected_lines_serial(F const& f, PVRow b = PVROW_INVALID_VALUE, const PVRow a = 0) const
	{
		if (b == PVROW_INVALID_VALUE) {
			b = count();
		}

		if (!_table || (b <= 0)) {
			return;
		}

		b--;

		size_t chunk_start = line_index_to_chunk(a);
		const size_t chunk_end = line_index_to_chunk(b);

		const PVRow cbit = line_index_to_chunk_bit(a);
		if (cbit > 0) {
			// Prelogue
			uint64_t cv = _table[chunk_start];
			PVRow end_bit = CHUNK_SIZE - 1;
			bool same = (chunk_end == chunk_start);
			if (same) {
				end_bit = line_index_to_chunk_bit(b);
			}

			const PVRow offset = chunk_to_line_index(chunk_start);
			for (PVRow b = cbit; b <= end_bit; b++) {
				if ((cv & (1ULL << b)) != 0) {
					f(b + offset);
				}
			}
			if (same) {
				return;
			}
			chunk_start++;
		}

		// Main loop
		size_t c;
		for (c = chunk_start; c < chunk_end; c++) {
			PVCore::PVBitVisitor::visit_bits(_table[c], f, chunk_to_line_index(c));
		}

		uint64_t cv = _table[chunk_end];
		const PVRow end_bit = line_index_to_chunk_bit(b);
		const PVRow offset = chunk_to_line_index(chunk_end);
		for (PVRow b = 0; b <= end_bit; b++) {
			if ((cv & (1ULL << b)) != 0) {
				f(b + offset);
			}
		}
	}

  protected:
	void allocate_table();
	void free_table();

	inline void allocate_and_copy_from(PVSelBitField const& o)
	{
		if (o._table) {
			allocate_table();
			copy_from(o);
		} else {
			if (_table) {
				free_table();
			}
			_table = NULL;
		}
	}

	void copy_from(PVSelBitField const& o);

  protected:
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/);
	ssize_t get_min_last_nonzero_chunk_index(PVSelBitField const& other) const;
	ssize_t get_max_last_nonzero_chunk_index(PVSelBitField const& other) const;

  protected:
	pointer _table;
	pvcop::core::memarray<bool>* _selection = nullptr;
	PVRow _count;
};
}

#endif
