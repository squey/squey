//-----------------------------------------------------------------------------
// Copyright © 2003 - Philip Howard - All rights reserved
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
//-----------------------------------------------------------------------------
// package	vrb
// homepage	http://vrb.slashusr.org/
//-----------------------------------------------------------------------------
// author	Philip Howard
// email	vrb at ipal dot org
// homepage	http://phil.ipal.org/
//-----------------------------------------------------------------------------
// This file is best viewed using a fixed spaced font such as Courier
// and in a display at least 120 columns wide.
//-----------------------------------------------------------------------------

#include <limits.h>
#include <stdio.h>
#include <unistd.h>

int main( int argc, char **argv ) {
    char symdata[ PATH_MAX+1 ];

    while ( ++ argv, -- argc > 0 ) {
	ssize_t len;
	len = (ssize_t) readlink( * argv, symdata, sizeof symdata );
	if ( len >= 0 ) {
	    symdata[ len ] = 0;
	    fputs( symdata, stdout );
	    fputc( '\n', stdout );
	    fflush( stdout );
	}
    }
    return 0;
}
