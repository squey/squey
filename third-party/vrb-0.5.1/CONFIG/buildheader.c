//-----------------------------------------------------------------------------
// Copyright © 2004 - Philip Howard - All rights reserved
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

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

int main( int argc, char **argv ) {
    static char *pass_list[] = {
	"__PREFIX_BEGIN__",	"__PREFIX_END__",
	"__INCLUDE_BEGIN__",	"__INCLUDE_END__",
	"__CONFIG_BEGIN__",	"__CONFIG_END__",
	"__DEFINE_BEGIN__",	"__DEFINE_END__",
	"__MACRO_BEGIN__",	"__MACRO_END__",
	"__FMACRO_BEGIN__",	"__FMACRO_END__",
	"__INLINE_BEGIN__",	"__INLINE_END__",
	"__PROTO_BEGIN__",	"__PROTO_END__",
	"__ALIAS_BEGIN__",	"__ALIAS_END__",
	"__SUFFIX_BEGIN__",	"__SUFFIX_END__",
	NULL };
    FILE *infile,*outfile;
    char **fname_ptr,**begin_ptr,**end_ptr,*line_ptr;
    size_t begin_len,end_len;
    int fname_cnt,do_copy;
    char line_buf[4096];

    if ( -- argc <= 0 ) {
	fprintf( stderr, "buildheader: no output file name\n" );
	return 1;
    }
    ++ argv;
    if ( ! ( outfile = fopen( *argv, "w" ) ) ) {
	fprintf( stderr, "buildheader: error opening \"%s\" for write: %s\n", *argv, strerror( errno ) );
	return 1;
    }
    begin_ptr = pass_list;
    while ( *begin_ptr ) {
	end_ptr = begin_ptr + 1;
	begin_len = strlen( *begin_ptr );
	end_len = strlen( *end_ptr );
	fname_ptr = argv;
	fname_cnt = argc;
	while ( ++ fname_ptr, -- fname_cnt > 0 ) {
	    if ( ! ( infile = fopen( *fname_ptr, "r" ) ) ) {
		fprintf( stderr, "buildheader: error opening \"%s\" for read: %s\n", *fname_ptr, strerror( errno ) );
		continue;
	    }
	    do_copy = 0;
	    while ( fgets( line_buf, sizeof line_buf, infile ) ) {
		line_ptr = line_buf;
		while ( *line_ptr && isspace( *line_ptr ) ) ++ line_ptr;
		if ( memcmp( line_ptr, *begin_ptr, begin_len ) == 0 ) do_copy = 1;
		else if ( memcmp( line_ptr, *end_ptr, end_len ) == 0 ) {
		    do_copy = 0;
		    if ( strcmp( "__PROTO_END__", *end_ptr ) == 0 ) fputs( ";\n\n", outfile );
		}
		else if ( do_copy ) fputs( line_buf, outfile );
	    }
	    fclose( infile );
	}
	begin_ptr = end_ptr + 1;
    }
    fclose( outfile );
    return 0;
}
