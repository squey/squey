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
// function	vrb_resize
//
// purpose	Change the size of the buffer space in a VRB.
//
// arguments	1 (vrb_p) pointer to vrb to resize
//		2 (size_t) new size, which will be rounded up as needed
//		3 (const char *) filename to memory map into
//
// returns	(int) == 0 : OK
//		(int)  < 0 : error
//
// note		This operation uses a lot of resources, so it should be used
//		only sparingly.
//-----------------------------------------------------------------------------
int
vrb_resize (
    vrb_p		arg_vrb
    ,
    size_t		arg_size
    ,
    const char *	arg_name
    )
    __PROTO_END__
{
    struct vrb		vrb_one		;
    struct vrb		vrb_two		;

    if ( ! arg_vrb ) {
	errno = EINVAL;
	return -1;
    }

    //-- If requested size cannot hold data, give up.
    if ( vrb_data_len( arg_vrb ) > arg_size ) {
	errno = ENOSPC;
	return -1;
    }

    //-- Set up a temporary VRB to get a buffer.
    if ( ! vrb_init( & vrb_one, arg_size, arg_name ) ) {
	errno = EFAULT;
	return -1;
    }

    //-- Move all the data to the new buffer.
    if ( vrb_data_len( arg_vrb ) > 0 ) {
	vrb_move( & vrb_one, arg_vrb, ~0 );
    }

    //-- Swap buffers.
    memcpy( & vrb_two, arg_vrb, sizeof (struct vrb) );
    memcpy( arg_vrb, & vrb_one, sizeof (struct vrb) );

    //-- Trash the old space in a temporary VRB.
    if ( vrb_uninit( & vrb_two ) < 0 ) {
	errno = EFAULT;
	return -1;
    }

    return 0;
}
