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
// macro	vrb_read_min
//
// purpose	Read some data from a file descriptor into a vrb only if
//		enough space is available to read a specified minimum amount.
//
// arguments	1 (vrb_p) pointer to vrb
//		2 (int) file descriptor
//		3 (size_t) maximum size to read or ~0 for no limit
//		4 (size_t) minimum size to read
//
// returns	(size_t)  >  0 : size actually read into vrb
//		(size_t) ==  0 : insufficient space to read into
//		(size_t) == ~0 && errno == 0 : end of file in read()
//		(size_t) == ~0 && errno != 0 : error from read(), see errno
//
// note		The EOF return semantics are different than for read().
//-----------------------------------------------------------------------------
#define vrb_read_min(b,f,x,m) (((m)<(vrb_space_len((b))))?0:(vrb_read((b),(f),(x))))

//-----------------------------------------------------------------------------
// function	vrb_read
//
// purpose	Read some data from a file descriptor into a vrb.
//
// arguments	1 (vrb_p) pointer to vrb
//		2 (int) file descriptor
//		3 (size_t) maximum size to read or ~0 for no limit
//
// returns	(size_t)  >  0 : size actually read into vrb
//		(size_t) ==  0 : no space to read into
//		(size_t) == ~0 && errno == 0 : end of file in read()
//		(size_t) == ~0 && errno != 0 : error from read(), see errno
//
// note		The EOF return semantics are different than for read().
//-----------------------------------------------------------------------------
size_t
vrb_read (
    vrb_p        arg_vrb
    ,
    int		arg_fd
    ,
    size_t	arg_size
    )
    __PROTO_END__
{
    ssize_t	num_read	;

    //-- If no vrb is given, return an error.
    if ( ! arg_vrb ) {
	errno = EINVAL;
	return ~ (size_t) 0;
    }

    //-- Determine available space and limit request to what is there.
    if ( arg_size > ( vrb_space_len( arg_vrb ) ) ) {
	arg_size = vrb_space_len( arg_vrb );
    }
    if ( arg_size == 0 ) return 0;

    //-- Read data once.
    num_read = read( arg_fd, arg_vrb->last_ptr, arg_size );
    if ( num_read <= 0 ) {
	if ( num_read == 0 ) errno = 0;
	return ~ (size_t) 0;
    }

    //-- Adjust for amount read.
    arg_vrb->last_ptr += num_read;

    //-- Return amount read to caller.
    return (size_t) num_read;
}
