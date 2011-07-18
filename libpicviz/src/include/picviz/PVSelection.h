//! \file PVSelection.h
//! $Id: PVSelection.h 3114 2011-06-14 03:22:15Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVSELECTION_H
#define PICVIZ_PVSELECTION_H

#include <pvcore/stdint.h>
#include <picviz/general.h>

#include <vector>


namespace Picviz {

#define PICVIZ_SELECTION_CHUNK_SIZE 32
#define PICVIZ_SELECTION_NUMBER_OF_CHUNKS PICVIZ_LINES_MAX / PICVIZ_SELECTION_CHUNK_SIZE /* 262144*/
#define PICVIZ_SELECTION_NUMBER_OF_BYTES PICVIZ_LINES_MAX / 8

/* Selection for axes */
#define PICVIZ_SELECTION_AXES_NUMBER_OF_CHUNKS PICVIZ_AXES_MAX / PICVIZ_SELECTION_CHUNK_SIZE
#define PICVIZ_SELECTION_AXES_NUMBER_OF_BYTES PICVIZ_AXES_MAX / 8


/**
* \class PVSelection
*/
class LibExport PVSelection {
private:
	std::vector<uint32_t> table;

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

	/**
	 * Destructor
	 */

	void* get_buffer();

	/**
	 * Get the state of line N in the PVSelection
	 *
	 * @param line_index The index of the line we are interested in
	 *
	 * @return A boolean stating whether it is set or not
	 */
	bool get_line (pv_row line_index) const;

	/**
	 * Gets the number of lines that are selected, in the range [a,b[
	 * So b is NOT INCLUDED !
	 *
	 * @param a The index of the first line in the range
	 * @param b The index of the line after the last one in the range (NOT INCLUDED !)
	 *
	 * @return The number of selected line in that range
	 */
	int get_number_of_selected_lines_in_range(pv_row a, pv_row b);

	/**
	 * This is the copy operator
	 * B = A
	 *
	 * @param &rhs
	 *
	 * @return The resulting PVSelection
	 */
	//PVSelection & operator=(const PVSelection &rhs);

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
	PVSelection & operator&=(const PVSelection &rhs);

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
	PVSelection & operator|=(const PVSelection &rhs);

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

	void select_all();
	void select_even();
	void select_none();
	void select_odd();

	/**
	 * Sets the state of line N in the PVSelection
	 *
	 * @param line_index The index of the line we are interested in
	 * @param bool_value The boolean value we want to set for that line_index
	 *
	 */
	void set_line(PVRow line_index, bool bool_value);

	inline void set_bit_fast(pv_row line_index){table[line_index / PICVIZ_SELECTION_CHUNK_SIZE] |= 1 << (line_index % PICVIZ_SELECTION_CHUNK_SIZE);}

	/**
	 * Get the float table from PVSelection.
	 */
         std::vector<PVRow> get_rows_table();


};
}

#endif /* PICVIZ_PVSELECTION_H */
