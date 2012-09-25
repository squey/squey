/**
 * \file PVSelection.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PICVIZ_PVSELECTION_H
#define PICVIZ_PVSELECTION_H

#include <pvkernel/core/stdint.h>
#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/PVAllocators.h>
#include <pvkernel/core/PVBitVisitor.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/core/PVAlignedBlockedRange.h>

#include <picviz/general.h>

#include <tbb/parallel_for.h>

#include <QTextStream>

#include <vector>

namespace PVRush {
class PVNraw;
}

namespace Picviz {

class PVSparseSelection;

#define PICVIZ_SELECTION_CHUNK_SIZE 64
#if (PICVIZ_LINES_MAX % PICVIZ_SELECTION_CHUNK_SIZE == 0)
#define PICVIZ_SELECTION_NUMBER_OF_CHUNKS (PICVIZ_LINES_MAX / PICVIZ_SELECTION_CHUNK_SIZE)
#else
#define PICVIZ_SELECTION_NUMBER_OF_CHUNKS ((PICVIZ_LINES_MAX / PICVIZ_SELECTION_CHUNK_SIZE) + 1)
#endif
#define PICVIZ_SELECTION_NUMBER_OF_ROWS (PICVIZ_SELECTION_NUMBER_OF_CHUNKS * PICVIZ_SELECTION_CHUNK_SIZE)
#define PICVIZ_SELECTION_NUMBER_OF_BYTES (PICVIZ_SELECTION_NUMBER_OF_ROWS / 8)


/* Selection for axes */
#define PICVIZ_SELECTION_AXES_NUMBER_OF_CHUNKS (PICVIZ_AXES_MAX / PICVIZ_SELECTION_CHUNK_SIZE + 1)

/**
* \class PVSelection
*/
class LibPicvizDecl PVSelection {
	friend class PVCore::PVSerializeObject;

public:
	//typedef PVCore::PVAlignedAllocator<uint32_t, 16>::pointer pointer;
	//typedef uint32_t DECLARE_ALIGN(16) * pointer;
	typedef uint64_t DECLARE_ALIGN(16) * pointer;
	typedef uint64_t DECLARE_ALIGN(16) const* const_pointer;
	typedef PVCore::PVAlignedAllocator<uint64_t, 16> allocator;

private:
	pointer _table;

public:
	/**
	 * Constructor
	 */
	PVSelection();

	/**
	 * Create a PVSelection object from a row table until we reach '0' marking
	 * the end of this row table.
	 *
	 * The PVSelection is a binary '0100' if the second line is selected.
	 */
	PVSelection(std::vector<PVRow> const& rtable);

	PVSelection(PVSelection const& o);

	~PVSelection()
	{
		if (_table) {
			free_table();
		}
	}

	// Move constructor. Save a lot of useless allocations, memcpys and desallocations !
	PVSelection(PVSelection&& o)
	{
		_table = o._table;
		o._table = NULL;
	}

	/**
	 * Destructor
	 */

	inline pointer get_buffer() { return _table; }
	inline const_pointer get_buffer() const { return _table; };

	/**
	 * Get the state of line N in the PVSelection
	 *
	 * @param line_index The index of the line we are interested in
	 *
	 * @return A boolean stating whether it is set or not
	 */
	inline bool get_line (PVRow line_index) const
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

		return (_table[pos] & (1UL<<shift));
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
	int get_number_of_selected_lines_in_range(PVRow a, PVRow b) const;

	bool is_empty() const;

	/**
	 * This is the copy operator
	 * B = A
	 *
	 * @param &rhs
	 *
	 * @return The resulting PVSelection
	 */
	PVSelection & operator=(const PVSelection &rhs);

	PVSelection& operator=(PVSelection&& rhs)
	{
		if (this != &rhs) {
			if (_table) {
				free_table();
			}
			_table = rhs._table;
			rhs._table = NULL;
		}

		return *this;
	}

	/**
	 * This is the binary outplaced 'AND' operation on two selections
	 * C = A & B
	 *
	 * @param rhs The second selection involved in the 'AND' operator
	 *
	 * @return The resulting PVSelection
	 */
	PVSelection operator&(const PVSelection &rhs) const;

	/**
	 * This is the binary INPLACED 'AND' operation on two selections
	 * A &= B  (that stands for A = A & B)
	 *
	 * @param rhs The second selection involved in the 'AND' operator
	 *
	 * @return A reference to the resulting PVSelection
	 */
	PVSelection& operator&=(const PVSelection &rhs);
	PVSelection& and_optimized(const PVSelection& rhs);

	/**
	 * This is the unary bitwise outplaced 'NOT' operation on one selection
	 * B = ~A
	 *
	 * @return The resulting PVSelection
	 */
	PVSelection operator~() const;

	/**
	 * This is the binary outplaced 'OR' operation on two selections
	 * C = A | B
	 *
	 * @param rhs The second selection involved in the 'OR' operator
	 *
	 * @return The resulting PVSelection
	 */
	PVSelection operator|(const PVSelection &rhs) const;

	/**
	 * This is the binary INPLACED 'OR' operation on two selections
	 * A |= B  (that stands for A = A | B)
	 *
	 * @param rhs The second selection involved in the 'OR' operator
	 *
	 * @return A reference to the resulting PVSelection
	 */
	PVSelection& operator|=(const PVSelection &rhs);
	PVSelection& operator|=(const PVSparseSelection &rhs);
	PVSelection& or_optimized(const PVSelection &rhs);

	/**
	 * This is the binary outplaced 'SUBSTRACTION' operation on two selections
	 * C = A - B
	 *
	 * @param rhs The second selection involved in the 'SUBSTRACTION' operator
	 *
	 * @return The resulting PVSelection
	 */
	PVSelection operator-(const PVSelection &rhs) const;

	/**
	 * This is the binary INPLACED 'SUBSTRACTION' operation on two selections
	 * A -= B  (that stands for A = A - B)
	 *
	 * @param rhs The second selection involved in the 'SUBSTRACTION' operator
	 *
	 * @return A reference to the resulting PVSelection
	 */
	PVSelection & operator-=(const PVSelection &rhs);

	void AB_sub(PVSelection const& a, PVSelection const& b);

	/**
	 * This is the binary outplaced 'XOR' operation on two selections
	 * C = A ^ B
	 *
	 * @param rhs The second selection involved in the 'XOR' operator
	 *
	 * @return The resulting PVSelection
	 */
	PVSelection operator^(const PVSelection &rhs) const;

	/**
	 * This is the binary INPLACED 'XOR' operation on two selections
	 * A ^= B  (that stands for A = A ^ B)
	 *
	 * @param rhs The second selection involved in the 'XOR' operator
	 *
	 * @return A reference to the resulting PVSelection
	 */
	PVSelection & operator^=(const PVSelection &rhs);

	/**
	 * This is the binary INPLACED 'OR NOT' operation on two selections
	 * A.or_not(B)  (that stands for A = A | ~B)
	 *
	 * @param rhs The second selection involved in the 'OR NOT' operator
	 *
	 * @return A reference to the resulting PVSelection
	 */
	PVSelection & or_not(const PVSelection &rhs);

	/**
	 * This is the binary INPLACED 'AND NOT' operation on two selections
	 * A.and_not(B)  (that stands for A = A | ~B)
	 *
	 * @param rhs The second selection involved in the 'AND NOT' operator
	 *
	 * @return A reference to the resulting PVSelection
	 */
	PVSelection & and_not(const PVSelection &rhs);

	/**
	 * This is the binary INPLACED 'XOR NOT' operation on two selections
	 * A.xor_not(B)  (that stands for A = A ^ ~B)
	 *
	 * @param rhs The second selection involved in the 'XOR NOT' operator
	 *
	 * @return A reference to the resulting PVSelection
	 */
	PVSelection & xor_not(const PVSelection &rhs);

	void select_all();
	void select_even();
	void select_none();
	void select_odd();
	void select_inverse();

	/**
	 * Sets the state of line N in the PVSelection
	 *
	 * @param line_index The index of the line we are interested in
	 * @param bool_value The boolean value we want to set for that line_index
	 *
	 */
	void set_line(PVRow line_index, bool bool_value);

	/**
	 * Sets the state of line N in the PVSelection only if bool_value is true
	 *
	 * @param line_index The index of the line we are interested in
	 * @param bool_value The boolean value we want to set for that line_index
	 *
	 */
	void set_line_select_only(PVRow line_index, bool bool_value);

	inline void set_bit_fast(PVRow line_index) { _table[line_index_to_chunk(line_index)] |= 1UL << (line_index_to_chunk_bit(line_index)); }

	// Returns the index of the chunk following the last chunk that contains a line
	// Thus, returns 0 if no chunk is empty
	ssize_t get_last_nonzero_chunk_index(ssize_t starting_chunk = 0, ssize_t ending_chunk = PICVIZ_SELECTION_NUMBER_OF_CHUNKS-1) const;

#if 0
	template <class F>
	void visit_selected_lines(F const& f, PVRow nrows = PICVIZ_SELECTION_NUMBER_OF_ROWS) const
	{
#ifdef __SSE_4_1__
		const ssize_t last_chunk = get_last_nonzero_chunk_index(line_index_to_chunk(nrows - 1));
		if (last_chunk == -1) {
			// No lines are selected !
			return;
		}
		__m128i sse_sel;
		const __m128i ones = _mm_set1_epi32(0xFFFFFFFF);
		const size_t last_chunk_sse = (((size_t)last_chunk)>>1)<<1;
		size_t i;
		for (i = 0; i < last_chunk_sse; i += 2) {
			sse_sel = _mm_load_si128((__m128i*) &_table[i]);
			if (_mm_testz_si128(sse_sel, ones) == 1) {
				// If this vector is null, then go one the next one.
				continue;
			}

			const uint64_t vec64_0 = _mm_extract_epi64(sse_sel, 0);
			const uint64_t vec64_1 = _mm_extract_epi64(sse_sel, 1);
			size_t nbits = _mm_popcnt_u64(vec64_0) + _mm_popcnt_u64(vec64_1);
			size_t cur_b = 0;
			const PVRow off = chunk_to_line_index(i);

			if (vec64_0 != 0) {
				while (nbits > 0 && cur_b < 64) {
					const PVRow idx = off + cur_b;
					if (idx >= nrows) {
						return;
					}
					if (vec64_0 & (1UL<<cur_b)) {
						f(idx);
						nbits--;
					}
					cur_b++;
				}
				cur_b = 0;
			}
			if (vec64_1 != 0) {
				while (nbits > 0 && cur_b < 64) {
					const PVRow idx = off + cur_b + 64;
					if (idx >= nrows) {
						return;
					}
					if (vec64_1 & (1UL<<cur_b)) {
						f(idx);
						nbits--;
					}
					cur_b++;
				}
			}
			assert(nbits == 0);
		}
		for (; i <= last_chunk; i++) {
			const uint64_t sel_buf = _table[i];
			for (PVRow j = 0; j < 64; j++) {
				const PVRow idx = chunk_to_line_index(i) +j;
				if (idx >= nrows) {
					return;
				}
				if (sel_buf & (1U<<j)) {
					f(idx);
				}
			}
		}
#else
		visit_selected_lines_serial(f, nrows);
#endif
	}
#endif

	template <class F>
	void visit_selected_lines(F const& f, const PVRow nrows = PICVIZ_SELECTION_NUMBER_OF_ROWS) const
	{
		if (!_table || (nrows <= 0)) {
			return;
		}
#ifdef __SSE4_1__
		const ssize_t last_chunk = get_last_nonzero_chunk_index(0, line_index_to_chunk(nrows - 1));
		if (last_chunk == -1) {
			// No lines are selected !
			return;
		}
		__m128i sse_sel;
		size_t last_chunk_sse = (((size_t)(last_chunk))>>1)<<1;
		size_t i;
		for (i = 0; i < last_chunk_sse; i += 2) {
			sse_sel = _mm_load_si128((__m128i*) &_table[i]);
			PVCore::PVBitVisitor::visit_bits(sse_sel, f, chunk_to_line_index(i));
		}
		for (; i < (size_t) last_chunk; i++) {
			const uint64_t sel_buf = _table[i];
			PVCore::PVBitVisitor::visit_bits(sel_buf, f, chunk_to_line_index(i));
		}
		// On last chunk, do not overflow !
		PVCore::PVBitVisitor::visit_bits(_table[last_chunk], [=,&f](const PVRow r){ if (r < nrows) f(r); }, chunk_to_line_index(last_chunk));
#else
		visit_selected_lines_serial(f, nrows);
#endif
	}

	template <class F>
	void visit_selected_lines_tbb(F const& f, const PVRow nrows = PICVIZ_SELECTION_NUMBER_OF_ROWS) const
	{
		if (!_table || (nrows <= 0)) {
			return;
		}
#ifdef __SSE4_1__
		const ssize_t last_chunk = get_last_nonzero_chunk_index(0, line_index_to_chunk(nrows - 1));
		if (last_chunk == -1) {
			// No lines are selected !
			return;
		}
		size_t last_chunk_sse = (((size_t)(last_chunk))>>1)<<1;
		size_t i;
		tbb::parallel_for(PVCore::PVAlignedBlockedRange<size_t, 2>(9, last_chunk_sse),
			[&](PVCore::PVAlignedBlockedRange<size_t, 2> const& range)
			{
				__m128i sse_sel;
				for (size_t i = range.begin(); i != range.end(); i++) {
					sse_sel = _mm_load_si128((__m128i*) &_table[i]);
					PVCore::PVBitVisitor::visit_bits(sse_sel, f, chunk_to_line_index(i));
				}
			});
		for (; i < last_chunk; i++) {
			const uint64_t sel_buf = _table[i];
			PVCore::PVBitVisitor::visit_bits(sel_buf, f, chunk_to_line_index(i));
		}
		// On last chunk, do not overflow !
		PVCore::PVBitVisitor::visit_bits(_table[last_chunk], [=,&f](const PVRow r){ if (r < nrows) f(r); }, chunk_to_line_index(last_chunk));
#else
		visit_selected_lines_serial(f, nrows);
#endif
	}

	/**
	 * Get the float table from PVSelection.
	 */
	std::vector<PVRow> get_rows_table();

	void write_selected_lines_nraw(QTextStream& stream, PVRush::PVNraw const& nraw, PVRow write_max);

	static inline PVRow line_index_to_chunk(const PVRow r) { return r>>6; }
	static inline PVRow chunk_to_line_index(const PVRow r) { return r<<6; }
	static inline PVRow line_index_to_chunk_bit(const PVRow r) { return r & 63; }

public:
#if 0
	template <class F>
	void visit_selected_lines_serial(F const& f, PVRow nrows = PICVIZ_SELECTION_NUMBER_OF_ROWS) const
	{
		const ssize_t last_chunk = get_last_nonzero_chunk_index(line_index_to_chunk(nrows - 1));
		if (last_chunk == 0) {
			return;
		}
		for (ssize_t i = 0; i < last_chunk; i++) {
			const uint64_t sel_buf = _table[i];
			if (sel_buf == 0) {
				continue;
			}
			for (uint32_t j = 0; j < 64; j++) {
				PVRow idx = chunk_to_line_index(i) + j;
				if (idx >= nrows) {
					return;
				}
				if (sel_buf & (1ULL << j)) {
					f(idx);
				}
			}
		}
	}
#endif

	template <class F>
	void visit_selected_lines_serial(F const& f, const PVRow nrows = PICVIZ_SELECTION_NUMBER_OF_ROWS) const
	{
		if (!_table || (nrows <= 0)) {
			return;
		}
		const ssize_t last_chunk = get_last_nonzero_chunk_index(0, line_index_to_chunk(nrows - 1));
		if (last_chunk == -1) {
			return;
		}
		for (ssize_t i = 0; i < last_chunk; i++) {
			const uint64_t sel_buf = _table[i];
			PVCore::PVBitVisitor::visit_bits(sel_buf, f, chunk_to_line_index(i));
		}
		// On last chunk, do not overflow !
		PVCore::PVBitVisitor::visit_bits(_table[last_chunk], [=,&f](const PVRow r){ if (r < nrows) f(r); }, chunk_to_line_index(last_chunk));
	}

private:
	inline void allocate_table() { _table = allocator().allocate(PICVIZ_SELECTION_NUMBER_OF_CHUNKS); }
	inline void free_table() { allocator().deallocate(_table, PICVIZ_SELECTION_NUMBER_OF_CHUNKS); }
	inline void allocate_and_copy_from(PVSelection const& o)
	{
		if (o._table) {
			allocate_table();
			copy_from(o);
		}
		else {
			_table = NULL;
		}
	}

	inline void copy_from(PVSelection const& o)
	{
		assert(_table);
		assert(o._table);
		static_assert(PICVIZ_SELECTION_NUMBER_OF_CHUNKS % 2 == 0, "PICVIZ_SELECTION_NUMBER_OF_CHUNKS must be a multiple of 2.");
		__m128i sse_c;
		for (size_t i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i += 2) {
			sse_c = _mm_load_si128((__m128i const*) &o._table[i]);
			_mm_store_si128((__m128i*) &_table[i], sse_c);
		}
		//memcpy(_table, o._table, PICVIZ_SELECTION_NUMBER_OF_BYTES);
	}

protected:
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/);
	ssize_t get_min_last_nonzero_chunk_index(PVSelection const& other) const;
	ssize_t get_max_last_nonzero_chunk_index(PVSelection const& other) const;
};

}

#endif /* PICVIZ_PVSELECTION_H */
