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
// function	vrb_take
//
// purpose	Specify how much data is taken from the specified buffer and
//		update pointers to reflect this change.
//
// arguments	1 (vrb_p) pointer to vrb
//		2 (size_t) size that has been taken
//
// returns	(int) == 0 : OK
//		(int)  < 0 : error
//-----------------------------------------------------------------------------
int
vrb_take (
    vrb_p	arg_vrb
    ,
    size_t	arg_size
    )
    __PROTO_END__
{
    //-- If no vrb is specified, return an error.
    if ( ! arg_vrb ) {
	errno = EINVAL;
	return -1;
    }

    //-- If too much is taken, then just return an error.
    if ( arg_size > vrb_data_len( arg_vrb ) ) {
	errno = ENOSPC;
	return -1;
    }

    //-- Adjust for amount taken.
    arg_vrb->first_ptr += arg_size;

    //-- If now in upper buffer, switch to lower buffer.
    if ( arg_vrb->first_ptr >= arg_vrb->upper_ptr ) {
	arg_vrb->first_ptr -= arg_vrb->buf_size;
	arg_vrb->last_ptr -= arg_vrb->buf_size;
    }

    return 0;
}
