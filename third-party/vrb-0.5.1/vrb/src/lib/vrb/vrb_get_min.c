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

#include "vrb_lib.h"

__PROTO_BEGIN__
//-----------------------------------------------------------------------------
// function	vrb_get_min
//
// purpose	Get a minimum amount of data from a vrb and copy it to a
//		caller provided space.	If the minimum request cannot be
//		fulfilled, nothing is copied.
//
// arguments	1 (vrb_p) pointer to vrb
//		2 (char *) pointer to where to copy data
//		3 (size_t) minimum size requested
//		4 (size_t) maximum size requested
//
// returns	(size_t) size of data copied to caller space
//		(size_t) ~0 : error
//-----------------------------------------------------------------------------
size_t
vrb_get_min (
    vrb_p	arg_vrb
    ,
    char *	arg_data
    ,
    size_t	arg_min_size
    ,
    size_t	arg_max_size
    )
    __PROTO_END__
{
    //-- If no vrb is given, return an error.
    if ( ! arg_vrb ) {
	errno = EINVAL;
	return ~0;
    }

    //-- Check minimum request validity.
    if ( arg_min_size > vrb_capacity( arg_vrb ) ) {
	errno = EINVAL;
	return ~0;
    }

    //-- Check minimum request for availability.
    if ( arg_min_size > vrb_data_len( arg_vrb ) ) {
	return 0;
    }

    //-- Fulfill request through vrb_get().
    return vrb_get( arg_vrb, arg_data, arg_max_size );
}
