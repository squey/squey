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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
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
// function	vrb_get
//
// purpose	Get data from a vrb and copy it to a caller provided space.
//
// arguments	1 (vrb_p) pointer to vrb
//		2 (char *) pointer to where to copy data
//		3 (size_t) size of data requested
//
// returns	(size_t) size of data copied to caller space
//		(size_t) ~0 : error
//-----------------------------------------------------------------------------
size_t
vrb_get (
    vrb_p	arg_vrb
    ,
    char *	arg_data
    ,
    size_t	arg_size
    )
    __PROTO_END__
{
    //-- If no vrb is given, return an error.
    if ( ! arg_vrb ) {
	errno = EINVAL;
	return ~0;
    }

    //-- Limit request to available data.
    if ( arg_size > vrb_data_len( arg_vrb ) ) {
	arg_size = vrb_data_len( arg_vrb );
    }

    //-- If nothing to get, then just return now.
    if ( arg_size == 0 ) return 0;

    //-- Copy data to caller space.
    memcpy( arg_data, arg_vrb->first_ptr, arg_size );

    //-- Adjust for amount copied.
    arg_vrb->first_ptr += arg_size;

    //-- If all the data was copied, then just reset pointers.
    if ( vrb_data_len( arg_vrb ) == 0 ) {
        arg_vrb->first_ptr = arg_vrb->lower_ptr;
        arg_vrb->last_ptr = arg_vrb->lower_ptr;
    }

    //-- If now in upper buffer, switch to lower buffer.
    else if ( arg_vrb->first_ptr >= arg_vrb->upper_ptr ) {
        arg_vrb->first_ptr -= arg_vrb->buf_size;
        arg_vrb->last_ptr -= arg_vrb->buf_size;
    }

    //-- Return amount copied to caller.
    return arg_size;
}
