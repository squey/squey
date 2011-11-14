#!/usr/bin/perl

my ($LOGFILE) = 0;

sub picviz_open_file {
    $filename = $_[0];
    open $main::LOGFILE, '<', $filename or die $!;
}

sub picviz_seek_begin {
	seek($main::LOGFILE,9,SEEK_SET)
}

sub picviz_pre_discovery {
	return 1;
}

sub picviz_is_element_text {
	return 1;
}

sub picviz_get_next_chunk() {
}


