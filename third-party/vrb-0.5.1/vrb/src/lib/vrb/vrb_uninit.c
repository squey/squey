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
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <unistd.h>

#include "vrb_lib.h"

__PROTO_BEGIN__
//-----------------------------------------------------------------------------
// function	vrb_uninit
//
// purpose	UN-initialize a vrb, releasing its buffer memory.
//
// arguments	1 (vrb_p) pointer to vrb to un-initialize
//
// returns	(int) == 0 : OK
//		(int)  < 0 : error
//-----------------------------------------------------------------------------
int
vrb_uninit (
    vrb_p	arg_vrb
    )
    __PROTO_END__
{
    if ( ! arg_vrb ) {
	errno = EINVAL;
	return -1;
    }

    if ( vrb_is_shmat( arg_vrb ) ) {
	shmdt( arg_vrb->lower_ptr );
	shmdt( arg_vrb->upper_ptr );
    } else if ( vrb_is_mmap( arg_vrb ) ) {
	munmap( arg_vrb->lower_ptr, arg_vrb->buf_size );
	munmap( arg_vrb->upper_ptr, arg_vrb->buf_size );
    } else {
	errno = EINVAL;
	return -1;
    }

    arg_vrb->lower_ptr	= (char *) NULL;
    arg_vrb->upper_ptr	= (char *) NULL;
    arg_vrb->first_ptr	= (char *) NULL;
    arg_vrb->last_ptr	= (char *) NULL;
    arg_vrb->buf_size	= 0;
    arg_vrb->flags	= 0;

    return 0;
}
