__PREFIX_BEGIN__
//-----------------------------------------------------------------------------
// Copyright © 2003 - Philip Howard - All rights reserved
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307	 USA
//-----------------------------------------------------------------------------
// package	vrb
// homepage	http://phil.ipal.org/freeware/vrb/
//-----------------------------------------------------------------------------
// author	Philip Howard
// email	vrb at ipal dot org
// homepage	http://phil.ipal.org/
//-----------------------------------------------------------------------------
// This file is best viewed using a fixed spaced font such as Courier
// and in a display at least 120 columns wide.
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// file		vrb.h
//
// purpose	Define resources used by and with the vrb library.
//-----------------------------------------------------------------------------
#ifndef __VRB_H__
#define __VRB_H__
__PREFIX_END__

__INCLUDE_BEGIN__
#include <stdlib.h>
__INCLUDE_END__

__DEFINE_BEGIN__
//-----------------------------------------------------------------------------
// struct	vrb
// typedef	VRB, vrb_p
//
// purpose	This is the vrb header structure used to hold the state
//		of this instance of a vrb.
//-----------------------------------------------------------------------------
struct vrb {
    char *			lower_ptr	;
    char *			upper_ptr	;
    char *			first_ptr	;
    char *			last_ptr	;
    char *			mem_ptr		;
    size_t			buf_size	;
    int				flags		;
};
typedef struct vrb		VRB		;
typedef VRB *			vrb_p		;

#define VRB_FLAG_SHMAT		0x0001
#define VRB_FLAG_MMAP		0x0002
#define VRB_FLAG_ERROR		0x0004
#define VRB_FLAG_GUARD		0x0008

#define vrb_flag_shmat(b)	(((b)->flags)&(VRB_FLAG_SHMAT))
#define vrb_flag_mmap(b)	(((b)->flags)&(VRB_FLAG_MMAP))
#define vrb_flag_error(b)	(((b)->flags)&(VRB_FLAG_ERROR))
#define vrb_flag_guard(b)	(((b)->flags)&(VRB_FLAG_GUARD))

#define vrb_is_shmat(b)		(!(!(((b)->flags)&(VRB_FLAG_SHMAT))))
#define vrb_is_mmap(b)		(!(!(((b)->flags)&(VRB_FLAG_MMAP))))
#define vrb_is_error(b)		(!(!(((b)->flags)&(VRB_FLAG_ERROR))))
#define vrb_is_guard(b)		(!(!(((b)->flags)&(VRB_FLAG_GUARD))))

#define vrb_is_not_shmat(b)	(!(((b)->flags)&(VRB_FLAG_SHMAT)))
#define vrb_is_not_mmap(b)	(!(((b)->flags)&(VRB_FLAG_MMAP)))
#define vrb_is_not_error(b)	(!(((b)->flags)&(VRB_FLAG_ERROR)))
#define vrb_is_not_guard(b)	(!(((b)->flags)&(VRB_FLAG_GUARD)))

#define vrb_set_shmat(b)	(((b)->flags)|=(VRB_FLAG_SHMAT))
#define vrb_set_mmap(b)		(((b)->flags)|=(VRB_FLAG_MMAP))
#define vrb_set_error(b)	(((b)->flags)|=(VRB_FLAG_ERROR))
#define vrb_set_guard(b)	(((b)->flags)|=(VRB_FLAG_GUARD))

#define vrb_unset_shmat(b)	(((b)->flags)&=(~(VRB_FLAG_SHMAT)))
#define vrb_unset_mmap(b)	(((b)->flags)&=(~(VRB_FLAG_MMAP)))
#define vrb_unset_error(b)	(((b)->flags)&=(~(VRB_FLAG_ERROR)))
#define vrb_unset_guard(b)	(((b)->flags)&=(~(VRB_FLAG_GUARD)))

//-----------------------------------------------------------------------------
// macros
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// macro	VRB_NOGUARD
//
// purpose	Option flag bit to disable guard pages
//-----------------------------------------------------------------------------
#define VRB_NOGUARD	0x0001

//-----------------------------------------------------------------------------
// macro	VRB_ENVGUARD
//
// purpose	Option flag bit to override VRB_NOGUARD when the environment
//		variable VRBGUARD is defined to a value other than "0"
//-----------------------------------------------------------------------------
#define VRB_ENVGUARD	0x0002

//-----------------------------------------------------------------------------
// macro	VRB_NOMKSTEMP
//
// purpose	Option flag bit to disable using mkstemp() to make a file name
//		from the given name pattern.  With this option, the name given
//		is used as an absolute name.
//-----------------------------------------------------------------------------
#define VRB_NOMKSTEMP	0x0004

//-----------------------------------------------------------------------------
// macro	vrb_capacity
//
// purpose	Return the capacity of the specified vrb.
//
// arguments	1 (vrb_p) pointer to the vrb to examine
//
// returns	(size_t) capacity of buffer
//-----------------------------------------------------------------------------
#define vrb_capacity(b) ((b)->buf_size)

//-----------------------------------------------------------------------------
// macro	vrb_data_len
//
// purpose	Return the current data length of the specified vrb.
//
// arguments	1 (vrb_p) pointer to the vrb to examine
//
// returns	(size_t) current data length in buffer
//-----------------------------------------------------------------------------
#define vrb_data_len(b) (((b)->last_ptr)-((b)->first_ptr))

//-----------------------------------------------------------------------------
// macro	vrb_data_ptr
//
// purpose	Get the pointer to data contents in the specified buffer.
//
//		Along with vrb_data_len(), this allows the caller direct
//		access to the data contents to determine how much to take,
//		and to call vrb_take() to specify how much data is taken.
//
// arguments	1 (vrb_p) pointer to vrb
//
// returns	(char *) pointer to data in buffer
//
// note		The data pointer only has meaning if there is data in the
//		buffer.	 Use vrb_data_len() to get the length to determine
//		how much data may be examined.
//-----------------------------------------------------------------------------
#define vrb_data_ptr(b) ((b)->first_ptr)

//-----------------------------------------------------------------------------
// macro	vrb_data_end
//
// purpose	Get the pointer to the end of data contents in the specified
//		buffer.	 This can be used by the caller to test a moving
//		pointer that was started from vrb_data_ptr() to scan data.
//
//		Along with vrb_data_ptr(), the allows the caller direct
//		access to the data contents to determine how much to take,
//		and to call vrb_take() to specify how much data is taken.
//
// arguments	1 (vrb_p) pointer to vrb
//
// returns	(char *) pointer to end of data in buffer
//
// note		The end of data pointer is always one past the last byte of
//		data, and cannot be used directly except to compare another
//		data pointer.
//
// note		The value of vrb_data_end() will be the same as vrb_data_ptr()
//		when the buffer is empty.
//-----------------------------------------------------------------------------
#define vrb_data_end(b) ((b)->last_ptr)

//-----------------------------------------------------------------------------
// macro	vrb_space_len
//
// purpose	Return the length of available space in the specified vrb.
//
// arguments	1 (vrb_p) pointer to the vrb to examine
//
// returns	(size_t) length of available space
//-----------------------------------------------------------------------------
#define vrb_space_len(b) (((b)->first_ptr)+((b)->buf_size)-((b)->last_ptr))

//-----------------------------------------------------------------------------
// macro	vrb_space_ptr
//
// purpose	Get the pointer to empty space in the specified buffer.
//
//		Along with vrb_space_len(), this allows the caller direct
//		access to the empty space to place data into the buffer,
//		and to call vrb_give() to specify home data is given.
//
// arguments	1 (vrb_p) pointer to vrb
//
// returns	(char *) pointer to space in buffer
//
// note		The space pointer only has meaning if there is space in the
//		buffer.	 Use vrb_available() to get the length to determine
//		how much data may be put in the buffer.
//-----------------------------------------------------------------------------
#define vrb_space_ptr(b) ((b)->last_ptr)

//-----------------------------------------------------------------------------
// macro	vrb_space_end
//
// purpose	Get the pointer to the end of empty space in the specified
//		buffer.
//
//		Along with vrb_space_ptr(), this allows the caller direct
//		access to the empty space to place data into the buffer,
//		and to call vrb_give() to specify home data is given.
//
// arguments	1 (vrb_p) pointer to vrb
//
// returns	(char *) pointer to end of space in buffer
//
// note		The end of data pointer is always one past the last byte of
//		space, and cannot be used directly except to compare another
//		space pointer.
//-----------------------------------------------------------------------------
#define vrb_space_end(b) (((b)->first_ptr)+((b)->buf_size))

//-----------------------------------------------------------------------------
// macro	vrb_is_empty
//
// purpose	Return true if the specified vrb is currently empty, else
//		return false.
//
// arguments	1 (vrb_p) pointer to the vrb to examine
//
// returns	(int) false : buffer is not empty
//		(int) true  : buffer is empty
//-----------------------------------------------------------------------------
#define vrb_is_empty(b) (((b)->last_ptr)==((b)->first_ptr))

//-----------------------------------------------------------------------------
// macro	vrb_is_full
//
// purpose	Return true if the specified vrb is currently full, else
//		return false.
//
// arguments	1 (vrb_p) pointer to the vrb to examine
//
// returns	(int) false : buffer is not full
//		(int) true  : buffer is full
//-----------------------------------------------------------------------------
#define vrb_is_full(b) (vrb_space_len((b))==0)

//-----------------------------------------------------------------------------
// macro	vrb_is_not_empty
//
// purpose	Return true if the specified vrb is currently empty, else
//		return false.
//
// arguments	1 (vrb_p) pointer to the vrb to examine
//
// returns	(int) false : buffer is not empty
//		(int) true  : buffer is empty
//-----------------------------------------------------------------------------
#define vrb_is_not_empty(b) (!(vrb_is_empty((b))))

//-----------------------------------------------------------------------------
// macro	vrb_is_not_full
//
// purpose	Return true if the specified vrb is currently full, else
//		return false.
//
// arguments	1 (vrb_p) pointer to the vrb to examine
//
// returns	(int) false : buffer is not full
//		(int) true  : buffer is full
//-----------------------------------------------------------------------------
#define vrb_is_not_full(b) (!(vrb_is_full((b))))
__DEFINE_END__

__SUFFIX_BEGIN__
#endif /* __VRB_H__ */
__SUFFIX_END__
