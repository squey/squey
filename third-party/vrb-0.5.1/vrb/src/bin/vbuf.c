//-----------------------------------------------------------------------------
// Copyright © 2006 - Philip Howard - All rights reserved
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

//-----------------------------------------------------------------------------
// program	vbuf
//
// purpose	Buffer data between programs in a pipeline with a progress
//		display showing byte count and transfer rate.
//
// syntax	vbuf [options]
//
// usage	producerprocess | vbuf | consumerprocess
//
//		-h	  Show this help message
//		-p	  Show progress display
//		-q	  Suppress progress display
//		-s size	  Set buffer size in bytes (default = 1m)
//			  (rounded up to system requirements)
//		-t time	  Set time in seconds between progress display
//			  refreshes (default = 2.5)
//		-b	  Show progress rate in bits per second
//		-B	  Show progress rate in bytes per second (default)
//		-d	  Show progress size in decimal (default)
//		-x	  Show progress size in hexadecimal
//
//		-i file	  Input from this file instead of stdin
//		-o file	  Output to this file instead of stdout
//
//		-M file   Memory map this file for the VRB
//
//		-r nnn	  Read a minimum of nnn bytes
//		-R nnn	  Read a maximum of nnn bytes
//
//		-w nnn	  Write a minimum of nnn bytes (until last)
//		-W nnn	  Write a maximum of nnn bytes
//-----------------------------------------------------------------------------
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <poll.h>
#include <sys/time.h>
#include <unistd.h>

#define _GNU_SOURCE
#include <getopt.h>

#include <vrb.h>

#define DEFAULT_DISPLAY_TIME_MS	1500
#define DEFAULT_BUFSIZE_PAGES   64
#define DEFAULT_BUFSIZE_MIN     0x00100000
#define DEFAULT_BUFSIZE_MAX     0x00800000
#define LIMIT_READ_MIN		0x01000000
#define LIMIT_WRITE_MIN		0x01000000
#define LIMIT_BUFSIZE_MIN	0x00001000
#define LIMIT_BUFSIZE_MAX       0x40000000

#ifndef EAGAIN
#define EAGAIN EWOULDBLOCK
#endif


//-----------------------------------------------------------------------------
// Define variables global to all functions in this program.
//-----------------------------------------------------------------------------
static const char *		program_name		= NULL;

static const struct option  long_opts       []      = {
    { "short-help",	no_argument,		NULL,	'h' },
    { "shorthelp",	no_argument,		NULL,	'h' },
    { "help",		no_argument,		NULL,	'H' },
    { "progress",	no_argument,		NULL,	'p' },
    { "quiet",		no_argument,		NULL,	'q' },
    { "bits",		no_argument,		NULL,	'b' },
    { "bytes",		no_argument,		NULL,	'B' },
    { "decimal",	no_argument,		NULL,	'd' },
    { "hexadecimal",	no_argument,		NULL,	'x' },
    { "cetal",		no_argument,		NULL,	'x' },
    { "size",		required_argument,	NULL,	's' },
    { "time",		required_argument,	NULL,	't' },
    { "input-file",	required_argument,	NULL,	'i' },
    { "inputfile",	required_argument,	NULL,	'i' },
    { "in-file",	required_argument,	NULL,	'i' },
    { "infile",		required_argument,	NULL,	'i' },
    { "output-file",	required_argument,	NULL,	'o' },
    { "outputfile",	required_argument,	NULL,	'o' },
    { "out-file",	required_argument,	NULL,	'o' },
    { "outfile",	required_argument,	NULL,	'o' },
    { "map-file",	required_argument,	NULL,	'M' },
    { "mapfile",	required_argument,	NULL,	'M' },
    { "read-minimum",	required_argument,	NULL,	'r' },
    { "readminimum",	required_argument,	NULL,	'r' },
    { "read-min",	required_argument,	NULL,	'r' },
    { "readmin",	required_argument,	NULL,	'r' },
    { "read-maximum",	required_argument,	NULL,	'R' },
    { "readmaximum",	required_argument,	NULL,	'R' },
    { "read-max",	required_argument,	NULL,	'R' },
    { "readmax",	required_argument,	NULL,	'R' },
    { "write-minimum",	required_argument,	NULL,	'w' },
    { "writeminimum",	required_argument,	NULL,	'w' },
    { "write-min",	required_argument,	NULL,	'w' },
    { "writemin",	required_argument,	NULL,	'w' },
    { "write-maximum",	required_argument,	NULL,	'W' },
    { "writemaximum",	required_argument,	NULL,	'W' },
    { "write-max",	required_argument,	NULL,	'W' },
    { "writemax",	required_argument,	NULL,	'W' },
    { NULL,		0,			NULL,	0 }
};

static const char	short_opts	[]	= ":?hHDpqbBdxs:t:i:o:M:r:R:w:W:";


//-----------------------------------------------------------------------------
// function	display_help
//
// purpose	Display help information.
//
// arguments	1 (int) help level, 1 for short, 2 for long
//
// returns	(void)
//-----------------------------------------------------------------------------
static
void
display_help (
    int			help_level
    )
{
    if ( help_level <= 1 ) {
	fprintf( stderr,
		 "syntax:       %s  [ options ]\n"
		 "options:\n"
		 "    -h        Show this help message\n"
		 "    -p        Show progress display\n"
		 "    -q        Suppress progress display\n"
		 "    -s size   Set buffer size in bytes (default = 1m)\n"
		 "              (rounded up to system requirements)\n"
		 "    -t time   Set time in seconds between progress display\n"
		 "              refreshes (default = %f)\n"
		 "    -b        Show progress rate in bits per second\n"
		 "    -B        Show progress rate in bytes per second (default)\n"
		 "    -d        Show progress size in decimal (default)\n"
		 "    -x        Show progress size in hexadecimal\n"
		 "    -i file   Input from this file instead of stdin\n"
		 "    -o file   Output to this file instead of stdout\n"
		 "    -M file   Memory map this file for the VRB\n"
		 "    -r nnn    Read a minimum of nnn bytes\n"
		 "    -R nnn    Read a maximum of nnn bytes\n"
		 "    -w nnn    Write a minimum of nnn bytes (until last)\n"
		 "    -W nnn    Write a maximum of nnn bytes\n",
		 program_name,
		 ( (double) DEFAULT_DISPLAY_TIME_MS ) / 1000.0 );
    } else {
	fprintf( stderr,
		 "\n"
		 "syntax:       %s  [ options ]\n"
		 "\n"
		 "usage:	producerprocess | %s | consumerprocess\n"
		 "\n"
		 "    -h\n"
		 "        Show short help message.\n"
		 "    --help\n"
		 "        Show long help message.\n"
		 "\n"
		 "    -p        --progress\n"
		 "        Show progress status line.\n"
		 "    -q        --quiet\n"
		 "        Suppress progress status line.\n"
		 "        The default is to show the progress display.\n"
		 "\n"
		 "    -s size   --size=size\n"
		 "        Set buffer size in bytes.  Suffixes k, m, and g may be used.\n"
		 "        The actual size used will be rounded up for system mapping\n"
		 "        requirements.  The default is 1m, or 1048576 bytes.\n"
		 "\n"
		 "    -t time   --time=seconds\n"
		 "        Set time in seconds between progress display refreshes.\n"
		 "        Fractions of a second may be used.  Default is %f seconds.\n"
		 "\n"
		 "    -b        --bits\n"
		 "        Set progress rate display units to bits per second.\n"
		 "    -B        --bytes\n"
		 "        Set progress rate display units to bytes per second.\n"
		 "        The default is bytes.\n"
		 "\n"
		 "    -d        --decimal\n"
		 "        Set progress rate display base to decimal.\n"
		 "        The default is decimal.\n"
		 "    -x        --hexadecimal\n"
		 "        Set progress rate display base to hexadecimal.\n"
		 "    --octal\n"
		 "        Set progress rate display base to octal.\n"
		 "\n"
		 "    -i file   --input=file\n"
		 "        Specify an input file to be opened and used instead of using\n"
		 "        stdin.\n"
		 "    -o file   --output=file\n"
		 "        Specify an output file to be opened and used instead of using\n"
		 "        stdout.\n"
		 "\n"
		 "    -M file   --mapfile=file\n"
		 "        Specify a file which will be used for backing store for the\n"
		 "        virtual ring buffer.  The allows using filesystem space in\n"
		 "        case swap space is full.\n"
		 "\n"
		 "    -r bytes  --read-min=bytes\n"
		 "        Specify the minimum number of bytes to be requested by read.\n"
		 "        Reading will not happen unless at least this much buffer space\n"
		 "        is available.  The default is 1.\n"
		 "    -R bytes  --read-max=bytes\n"
		 "        Specify the maximum number of bytes to be requested by read.\n"
		 "        Reading will not request more even if more buffer space is\n"
		 "        available.  The default is the full buffer size.\n"
		 "    -w bytes  --write-min=bytes\n"
		 "        Specify the minumum number of bytes to be written.  Writing\n"
		 "        will not be done unless at least this much data is available.\n"
		 "        Once end of file is reached on input, a smaller amount may be\n"
		 "        written.  The default is 1.\n"
		 "    -W bytes  --write-max=bytes\n"
		 "        Specify the maximum number of bytes to be written.  Writing\n"
		 "        will not write more each time even if more data is available\n"
		 "        in the buffer.  The default is the full buffer size.\n"
		 "\n",
		 program_name,
		 program_name,
		 ( (double) DEFAULT_DISPLAY_TIME_MS ) / 1000.0 );
    }
    return;
}

//-----------------------------------------------------------------------------
// function	display_status
//
// purpose	Format and display a progress status line.
//
// arguments	1 (double) full byte rate per second
//		2 (double) short byte rate per second
//		3 (size_t) total capacity of buffer
//		4 (size_t) quantity of data in buffer
//		5 (uint64_t) total write count
//		6 (int) display base, 10 or 16
//		7 (int) display unit, 1 (bits) or 8 (bytes)
//
// returns	(void)
//-----------------------------------------------------------------------------
static
void
display_status (
    double		full_rate
    ,
    double		short_rate
    ,
    uint64_t		write_count
    ,
    size_t		buffer_cap
    ,
    size_t		buffer_len
    ,
    int			display_base
    ,
    int			display_unit
    )
{
    static const char	decimal_prog[]		=
	"\r%14llu [%*lu]   %8.3f %c%cps  (%8.3f %c%cps) ";
    static const char	decimal_stat[]		=
	"\r%14llu [%*lu]   %8.3f %c%cps  (%9.3f sec)";

    static const char	hexadecimal_prog[]	=
	"\r%14llx [%*lx]   %8.3f %c%cps  (%8.3f %c%cps) ";
    static const char	hexadecimal_stat[]	=
	"\r%14llx [%*lx]   %8.3f %c%cps  (%9.3f sec)";

    static const char	suffixes	[] = " kMGTP";

    static int		old_full_scale	= 0;
    static int		old_short_scale	= 0;

    const char *	format		;

    unsigned long	cap_size	;
    int			fmt_size	;
    int			full_scale	;
    int			short_scale	;
    int			unit_ch		;


    unit_ch = 'B';
    if ( display_unit == 1 ) {
	unit_ch = 'b';
	full_rate *= 8.0;
	short_rate *= 8.0;
    }

    full_scale = 0;
    while ( full_scale < 5 ) {
	if ( full_rate < ( ( full_scale == old_full_scale ) ? 10000.0 : 1000.0 ) ) break;
	full_rate /= 1000.0;
	++ full_scale;
    }

    short_scale = 0;
    if ( short_rate < 0.0 ) {
	format = display_base == 16 ? hexadecimal_stat : decimal_stat;
	short_rate = - short_rate / 1000.0;
    } else {
	format = display_base == 16 ? hexadecimal_prog : decimal_prog;
	while ( short_scale < 5 ) {
	    if ( short_rate < ( ( short_scale == old_short_scale ) ? 10000.0 : 1000.0 ) ) break;
	    short_rate /= 1000.0;
	    ++ short_scale;
	}
    }

    if ( display_base == 16 )
	for ( fmt_size = 8, cap_size = 0x10000000; cap_size > buffer_cap; fmt_size -= 1, cap_size /= 16 );
    else
	for ( fmt_size = 10, cap_size = 1000000000; cap_size > buffer_cap; fmt_size -= 1, cap_size /= 10 );

    fprintf( stderr, format,
	     write_count, fmt_size, (unsigned long) buffer_len,
	     full_rate, suffixes[full_scale], unit_ch,
	     short_rate, suffixes[short_scale], unit_ch );

    old_full_scale = full_scale;
    old_short_scale = short_scale;

    return;
}

//-----------------------------------------------------------------------------
// macro	get_time_ms
//
// purpose	Get the current time in microseconds as a 64 bit integer.
//
// arguments	-none-
//
// returns	(timems_t) current time in microseconds
//-----------------------------------------------------------------------------
#define get_time_ms() ({							\
	long long this__time;							\
	struct timeval this__timeval;						\
	gettimeofday( & this__timeval, NULL );					\
	this__time = this__timeval.tv_sec;					\
	this__time *= 1000000LL;						\
	this__time += this__timeval.tv_usec;					\
	this__time /= 1000LL;							\
	this__time;								\
})

//-----------------------------------------------------------------------------
// function	main
//
// purpose	It all starts here.
//-----------------------------------------------------------------------------
int
main (
    int		argc
    ,
    char * *	argv
    ,
    char * *	envp
    )
{
    struct pollfd	poll_list		[2];

    double		delta			;
    double		rate			;
    double		smooth_rate		;
    double		smoothing		;
    double		max_smoothing		;

    uint64_t		write_count		;
    uint64_t		read_count		;
    uint64_t		old_write_count		;

    int64_t		display_time		;
    int64_t		start_time		;
    int64_t		previous_time		;

    vrb_p		io_buf			;

    const char *	opt_input		;
    const char *	opt_output		;
    const char *	opt_mapfile		;

    size_t		page_size		;
    size_t		opt_buf_size		;
    size_t		opt_read_min		;
    size_t		opt_read_max		;
    size_t		opt_write_min		;
    size_t		opt_write_max		;

    int			error_count		;
    int			input_fd		;
    int			opt_display_base	;
    int			opt_display_time	;
    int			opt_display_unit	;
    int			opt_help		;
    int			opt_progress		;
    int			output_fd		;
    int			poll_time		;
    int			poll_num		;
    int			poll_write		;
    int			poll_read		;


    //----------------------------
    // Do various initializations.
    //----------------------------
#ifdef _SC_PAGESIZE
    page_size = sysconf( _SC_PAGESIZE );
#else
    page_size = getpagesize();
#endif
    opt_buf_size = DEFAULT_BUFSIZE_PAGES * page_size;
    if ( opt_buf_size < DEFAULT_BUFSIZE_MIN ) opt_buf_size = DEFAULT_BUFSIZE_MIN;
    if ( opt_buf_size > DEFAULT_BUFSIZE_MAX ) opt_buf_size = DEFAULT_BUFSIZE_MAX;

    max_smoothing	= 8.0;
    smoothing		= 0.0;
    smooth_rate		= 0.0;

    opt_input		= NULL;
    opt_output		= NULL;
    opt_mapfile		= NULL;

    write_count		= 0;
    old_write_count	= 0;
    read_count		= 0;
    error_count		= 0;

    opt_help		= 0;
    opt_progress	= 1;
    opt_display_base	= 10;
    opt_display_time	= DEFAULT_DISPLAY_TIME_MS;
    opt_display_unit	= 8;

    opt_read_min	= 1;
    opt_write_min	= 1;

    opt_read_max	= 0;
    opt_write_max	= 0;

    input_fd		= -1;
    output_fd		= -1;

    poll_time		= -1;


    //-----------------------------------------
    // Extract the program name last component.
    //-----------------------------------------
    {
	char * p;
	program_name = p = argv[0];
	while ( * p ) if ( * p ++ == '/' && * p ) program_name = p;
    }

    //------------------------------------
    // Scan options from the command line.
    //------------------------------------
    for (;;) {
	int opt;

	opt = getopt_long( argc, argv, short_opts, long_opts, NULL );
	if ( opt < 0 ) break;

	switch ( opt ) {

	case 'H':
	    opt_help += 2;
	    break;

	case 'h':
	case '?':
	    ++ opt_help;
	    break;

	case 'p':
	    opt_progress = 1;
	    break;

	case 'q':
	    opt_progress = 0;
	    break;

	case 'b':
	    opt_display_unit = 1;
	    break;

	case 'B':
	    opt_display_unit = 8;
	    break;

	case 'd':
	    opt_display_base = 10;
	    break;

	case 'x':
	    opt_display_base = 16;
	    break;

	case 'O':
	    opt_display_base = 8;
	    break;

	case 's':
	    opt_buf_size = strtoul( optarg, NULL, 0 );
	    break;

	case 't':
	    opt_display_time = (int) ( 1000.0 * strtod( optarg, NULL ) );
	    break;

	case 'i':
	    opt_input = optarg;
	    break;

	case 'o':
	    opt_output = optarg;
	    break;

	case 'M':
	    opt_mapfile = optarg;
	    break;

	case 'r':
	    opt_read_min = strtoul( optarg, NULL, 0 );
	    break;

	case 'R':
	    opt_read_max = strtoul( optarg, NULL, 0 );
	    break;

	case 'w':
	    opt_write_min = strtoul( optarg, NULL, 0 );
	    break;

	case 'W':
	    opt_write_max = strtoul( optarg, NULL, 0 );
	    break;

	default:
	    fprintf( stderr, "%s: unknown option: %s\n", program_name, argv[ optind ] );
	    ++ error_count;
	}
    }

    if ( opt_help ) {
	display_help( opt_help );
	exit( 1 );
    }
    if ( error_count ) {
	exit( 1 );
    }

    //---------------------------------------------------
    // Make sure minimums are non-zero and not too large.
    //---------------------------------------------------
    if ( opt_read_min == 0 ) {
	opt_read_min = 1;
    }
    else if ( opt_read_min > LIMIT_READ_MIN ) {
	opt_read_min = LIMIT_READ_MIN;
    }
    if ( opt_write_min == 0 ) {
	opt_write_min = 1;
    }
    else if ( opt_write_min > LIMIT_WRITE_MIN ) {
	opt_write_min = LIMIT_WRITE_MIN;
    }

    //-------------------------------------------------------
    // Make sure buffer size is at least enough for minimums.
    //-------------------------------------------------------
    if ( opt_buf_size < ( opt_read_min + opt_write_min ) ) {
	opt_buf_size = opt_read_min + opt_write_min;
    }

    //----------------------------------------
    // Make sure buffer size is within range
    // and an exact multiple of the page size.
    //----------------------------------------
    if ( opt_buf_size < LIMIT_BUFSIZE_MIN ) {
	opt_buf_size = LIMIT_BUFSIZE_MIN;
    }
    if ( opt_buf_size > LIMIT_BUFSIZE_MAX ) {
	opt_buf_size = LIMIT_BUFSIZE_MAX;
    }
    opt_buf_size += page_size - 1;
    opt_buf_size -= opt_buf_size % page_size;

    //----------------------------------
    // Set up maximums if not specified.
    //----------------------------------
    if ( opt_read_max == 0 ) {
	opt_read_max = opt_buf_size;
    }
    if ( opt_write_max == 0 ) {
	opt_write_max = opt_buf_size;
    }

    //------------------------------
    // Create the buffer to be used.
    //------------------------------
    io_buf = vrb_new( opt_buf_size, opt_mapfile );
    if ( ! io_buf ) {
	fprintf( stderr, "%s: error creating buffer: %s\n", program_name, strerror( errno ) );
	return 1;
    }
    opt_buf_size = vrb_capacity( io_buf );

    //----------------------------------------
    // Touch the buffer space once.  Sometimes
    // read() can fail if this is not done.
    //----------------------------------------
    * (volatile unsigned long *) vrb_space_ptr( io_buf ) = 0;

    //--------------------------------------------------------
    // If an input file is specified, open it, else use STDIN.
    //--------------------------------------------------------
    if ( opt_input ) {
	input_fd = open( opt_input, O_RDONLY );
	if ( input_fd < 0 ) {
	    fprintf( stderr, "%s: error opening %sput: %s\n    file: %s\n",
		     program_name, "in", strerror( errno ), opt_input );
	    ++ error_count;
	}
    } else {
	opt_input = "[STDIN]";
	input_fd = STDIN_FILENO;
    }

    if ( error_count ) return 1;

    //----------------------------------------------------------
    // If an output file is specified, open it, else use STDOUT.
    //----------------------------------------------------------
    if ( opt_output ) {
	output_fd = open( opt_output, O_WRONLY | O_CREAT | O_TRUNC, 0644 );
	if ( output_fd < 0 ) {
	    fprintf( stderr, "%s: error opening %sput: %s\n    file: %s\n",
		     program_name, "out", strerror( errno ), opt_output );
	    ++ error_count;
	}
    } else {
	opt_output = "[STDOUT]";
	output_fd = STDOUT_FILENO;
    }

    if ( error_count ) return 1;

    //--------------------------------------
    // Set file descriptors to non-blocking.
    //--------------------------------------
    if ( fcntl( input_fd, F_SETFL, O_NONBLOCK ) < 0 ) {
	fprintf( stderr, "%s: error setting O_NONBLOCK on std%s: %s\n", "in", program_name, strerror( errno ) );
	++ error_count;
    }
    if ( fcntl( output_fd, F_SETFL, O_NONBLOCK ) < 0 ) {
	fprintf( stderr, "%s: error setting O_NONBLOCK on std%s: %s\n", "out", program_name, strerror( errno ) );
	++ error_count;
    }
    if ( error_count ) return 1;

    //----------------
    // Get start time.
    //----------------
    start_time = get_time_ms();
    previous_time = start_time;
    display_time = start_time;


    //---------------------------------------------
    // Initialize poll list with persistent values.
    //---------------------------------------------
    poll_list[0].fd = output_fd;
    poll_list[0].events = POLLOUT;
    poll_list[1].fd = input_fd;
    poll_list[1].events = POLLIN;

    //------------------------------------
    // Do main loop until all I/O is done.
    //------------------------------------
    poll_write = 0;
    poll_read = 0;
    for (;;) {

	//--------------------------------------------
	// If time to update progress display, do now.
	//--------------------------------------------
	if ( opt_progress ) {
	    int64_t	current_time	;

	    current_time = get_time_ms();
	    if ( current_time >= display_time ) {
		delta = ( (double) current_time ) - ( (double) previous_time );
		if ( delta == 0.0 ) {
		    rate = 0.0;
		    smooth_rate = 0.0;
		} else {
		    rate = ( (double) ( write_count - old_write_count ) ) * 1000.0 / delta;
		    if ( smoothing < max_smoothing ) smoothing += 1.0;
		    smooth_rate = ( smoothing * smooth_rate - smooth_rate + rate ) / smoothing;
		    
		    delta = ( (double) current_time ) - ( (double) start_time );
		    rate = ( (double) write_count ) * 1000.0 / delta;
		}

		display_status( rate, smooth_rate, write_count,
				opt_buf_size, vrb_data_len( io_buf ),
				opt_display_base, opt_display_unit );

		while ( display_time <= current_time ) {
		    display_time += opt_display_time;
		}
		old_write_count = write_count;
		previous_time = current_time;
	    }
	}

	//-------
	// Write.
	//-------
	if ( ! poll_write ) {
	    size_t	ulen		;
	    ssize_t	len		;

	    if ( ( ulen = vrb_data_len( io_buf ) ) >= opt_write_min ) {
		if ( ulen > opt_write_max ) ulen = opt_write_max;

		//-- Try writing.
		len = write( output_fd, vrb_data_ptr( io_buf ), ulen );

		//-- Handle an error return.
		if ( len < 0 ) {
		    if ( errno != EAGAIN && errno != EWOULDBLOCK ) {
			fprintf( stderr, "\n%s: %s( %d, %p, %lu ): %s\n", program_name, "write",
				 output_fd, vrb_data_ptr( io_buf ), (unsigned long) ulen, strerror( errno ) );
			break;
		    }
		    poll_write = 1;
		}

		//-- Handle a strange return.
		else if ( len == 0 ) {
		    fprintf( stderr, "%s: write error: %s\n", program_name, "returns zero" );
		    break;
		}

		//-- Handle a normal return.
		else {
		    vrb_take( io_buf, len );
		    write_count += len;
		    continue;
		}
	    }
	}

	//---------------------------------------
	// If EOF and buffer is empty, then quit.
	//---------------------------------------
	if ( input_fd < 0 ) {
	    if ( vrb_data_len( io_buf ) == 0 ) break;
	}

	//------
	// Read.
	//------
	else if ( ! poll_read ) {
	    size_t	ulen		;
	    ssize_t	len		;

	    if ( ( ulen = vrb_space_len( io_buf ) ) >= opt_read_min ) {
		if ( ulen > opt_read_max ) ulen = opt_read_max;

		//-- Try reading.
		len = read( input_fd, vrb_space_ptr( io_buf ), ulen );

		//-- Handle error returns.
		if ( len < 0 ) {
		    if ( errno != EAGAIN && errno != EWOULDBLOCK ) {
			fprintf( stderr, "\n%s: %s( %d, %p, %lu ): %s\n", program_name, "read",
				 input_fd, vrb_space_ptr( io_buf ), (unsigned long) ulen, strerror( errno ) );
			break;
		    }
		    poll_read = 1;
		}

		//-- Handle end of file.
		else if ( len == 0 ) {
		    close( input_fd );
		    input_fd = -1;
		    continue;
		}

		//-- Handle normal returns.
		else {
		    vrb_give( io_buf, len );
		    read_count += len;
		    continue;
		}
	    }
	}

	//--------------------------------------------
	// If nothing to poll, then we are deadlocked.
	//--------------------------------------------
	if ( ! poll_write && ! poll_read ) {
	    fprintf( stderr, "\n%s: DEADLOCK\n", program_name );
	    fprintf( stderr, "    buffer total size = %lu\n", (unsigned long) opt_buf_size );
	    fprintf( stderr, "    buffer space size = %lu\n", (unsigned long) vrb_space_len( io_buf ) );
	    fprintf( stderr, "    buffer data size  = %lu\n", (unsigned long) vrb_data_len( io_buf ) );
	    fprintf( stderr, "    read descriptor   = %d\n", input_fd );
	    fprintf( stderr, "    write descriptor  = %d\n", output_fd );
	    abort();
	}

	//------------------
	// Set up poll list.
	//------------------
	poll_list[0].fd = poll_write ? output_fd : -1;
	poll_list[0].events = poll_write ? POLLOUT : 0;
	poll_list[0].revents = 0;

	poll_list[1].fd = poll_read ? input_fd : -1;
	poll_list[1].events = poll_read ? POLLIN : 0;
	poll_list[1].revents = 0;

	//-------------------------------------------------------------
	// Wait in poll until I/O is ready or time to display progress.
	//-------------------------------------------------------------
	if ( opt_progress ) {
	    poll_time = display_time - get_time_ms();
	    if ( poll_time < 0 ) poll_time = 0;
	}
	poll_num = poll( poll_list, 2, poll_time );

	//-----------------------
	// Check for poll events.
	//-----------------------
	if ( poll_write && poll_list[0].revents ) poll_write = 0;
	if ( poll_read  && poll_list[1].revents ) poll_read  = 0;
    }

    //--------------------------------------------------
    // Do final progress display to reflect grand total.
    //--------------------------------------------------
    if ( opt_progress ) {
	delta = ( (double) get_time_ms() ) - ( (double) start_time );
	rate = ( (double) write_count ) * 1000.0 / delta;

	display_status( rate, -delta, write_count,
			opt_buf_size, vrb_data_len( io_buf ),
			opt_display_base, opt_display_unit );
	fputc( '\n', stderr );
    }

    //-------------------
    // Clean up and quit.
    //-------------------
    vrb_destroy( io_buf );
    close( output_fd );

    exit( 0 );
}
