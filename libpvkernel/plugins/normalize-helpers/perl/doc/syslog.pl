#!/usr/bin/perl

# \file syslog.pl
#
# Copyright (C) Picviz Labs 2010-2012

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
