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
// function	vrb_move
//
// purpose	Move data from one buffer to another buffer, limited only by
//		how much data there is in the source and how much space there
//		in the destination, or the specified amount.
//
// arguments	1 (vrb_p) pointer to destination vrb
//		2 (vrb_p) pointer to source vrb
//		3 (size_t) max length to move or ~0 for all
//
// returns	(size_t) size of data moved between buffers
//		(size_t) ~0 : error
//-----------------------------------------------------------------------------
size_t
vrb_move (
    vrb_p	arg_vrb_dst
    ,
    vrb_p	arg_vrb_src
    ,
    size_t	arg_max_len
    )
    __PROTO_END__
{
    size_t	move_len	;

    //-- If no vrb is given, return an error.
    if ( ! arg_vrb_dst || ! arg_vrb_src ) {
	errno = EINVAL;
	return ~0;
    }

    //-- Determine how much we can move.
    move_len = arg_max_len;
    if ( move_len > vrb_data_len( arg_vrb_src ) ) {
	move_len = vrb_data_len( arg_vrb_src );
    }
    if ( move_len > vrb_space_len( arg_vrb_dst ) ) {
	move_len = vrb_space_len( arg_vrb_dst );
    }

    //-- Move data by copying directly between buffers.
    memcpy( vrb_space_ptr( arg_vrb_dst ), vrb_data_ptr( arg_vrb_src ), move_len );

    //-- Update buffers to reflect the change.
    vrb_take( arg_vrb_src, move_len );
    vrb_give( arg_vrb_dst, move_len );

    //-- Return the length to the caller.
    return move_len;
}
