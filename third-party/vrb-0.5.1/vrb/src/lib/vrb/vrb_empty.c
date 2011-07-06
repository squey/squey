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
// function	vrb_empty
//
// purpose	Discard all the data in a buffer, making it empty.
//
// arguments	1 (vrb_p) pointer to vrb
//
// returns	(int) == 0 : OK
//		(int)  < 0 : error
//-----------------------------------------------------------------------------
int
vrb_empty (
    vrb_p	arg_vrb
    )
    __PROTO_END__
{
    //-- If no vrb is specified, return an error.
    if ( ! arg_vrb ) {
	errno = EINVAL;
	return -1;
    }

    //-- Reset pointers back to the beginning.
    arg_vrb->first_ptr = arg_vrb->lower_ptr;
    arg_vrb->last_ptr = arg_vrb->lower_ptr;

    return 0;
}
