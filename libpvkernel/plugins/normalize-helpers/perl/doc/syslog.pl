#!/usr/bin/perl
#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

my ($LOGFILE) = 0;

sub picviz_open_file {
    $filename = $_[0];
    open $main::LOGFILE, '<', $filename or die $!;
	print "From PERL: open_file\n";
}

sub picviz_seek_begin {
	print "From PERL: seek_begin\n";
	seek($main::LOGFILE,0,0);
}

sub picviz_pre_discovery {
	return 1;
}

sub picviz_is_element_text {
	return 1;
}

sub picviz_get_next_chunk {
	my ($min_chunk_size) = $_[0];
}
