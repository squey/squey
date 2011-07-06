//-----------------------------------------------------------------------------
// Copyright © 2002 - Philip Howard - All rights reserved
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

#include <stdio.h>
#include <unistd.h>

#include "vrb_lib.h"

__PROTO_BEGIN__
//-----------------------------------------------------------------------------
// macro	vrb_write_min
//
// purpose	Write some data from a vrb to a file descriptor only if
//		enough data is available to write a specified minimum amount.
//
// arguments	1 (vrb_p) pointer to vrb
//		2 (int) file descriptor
//		3 (size_t) maximum size to write or ~0 for no limit
//		4 (size_t) minimum size to write
//
// returns	(size_t)  >  0 : size actually written
//		(size_t) ==  0 : insufficient data in buffer
//		(size_t) == ~0 : error from write(), see errno
//-----------------------------------------------------------------------------
#define vrb_write_min(b,f,x,m) (((m)<(vrb_data_len((b))))?0:(vrb_write((b),(f),(x))))

//-----------------------------------------------------------------------------
// function	vrb_write
//
// purpose	Write some data from a vrb to a file descriptor.
//
// arguments	1 (vrb_p) pointer to vrb
//		2 (int) file descriptor
//		3 (size_t) maximum size to write or ~0 for no limit
//
// returns	(size_t)  >  0 : size actually written
//		(size_t) ==  0 : no data in buffer
//		(size_t) == ~0 : error from write(), see errno
//-----------------------------------------------------------------------------
size_t
vrb_write (
    vrb_p	arg_vrb
    ,
    int		arg_fd
    ,
    size_t	arg_size
    )
    __PROTO_END__
{
    ssize_t	num_write	;

    //-- If no vrb is given, return an error.
    if ( ! arg_vrb ) {
	errno = EINVAL;
	return ~0;
    }

    //-- Limit request to available data.
    if ( arg_size > vrb_data_len( arg_vrb ) ) {
	arg_size = vrb_data_len( arg_vrb );
    }
    if ( arg_size == 0 ) return 0;

    //-- Write data once.
    num_write = write( arg_fd, arg_vrb->first_ptr, arg_size );
    if ( num_write <= 0 ) {
	if ( num_write == 0 ) errno = 0;
	return ~0;
    }

    //-- Adjust for amount written.
    arg_vrb->first_ptr += num_write;

    //-- If all the data was written, then just reset pointers.
    if ( vrb_data_len( arg_vrb ) == 0 ) {
	arg_vrb->first_ptr = arg_vrb->lower_ptr;
	arg_vrb->last_ptr = arg_vrb->lower_ptr;
    }

    //-- If now in upper buffer, switch to lower buffer.
    else if ( arg_vrb->first_ptr >= arg_vrb->upper_ptr ) {
	arg_vrb->first_ptr -= arg_vrb->buf_size;
	arg_vrb->last_ptr -= arg_vrb->buf_size;
    }

    //-- Return amount written to caller.
    return (size_t) num_write;
}
