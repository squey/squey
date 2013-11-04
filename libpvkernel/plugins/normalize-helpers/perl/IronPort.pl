#!/usr/bin/perl

# \file IronPort.pl
#
# Copyright (C) Picviz Labs 2010-2012

# This script implements method useds by Picviz Inspector for the normalization process.
# Please note that processing files this way might be slower than using native picviz normalization process.

my ($LOGFILE) = 0;

# Open the file given as a parameter
sub picviz_open_file {
    $filename = $_[0];
    open $main::LOGFILE, '<', $filename or die $!;
}

# Seek the file to the beggining
sub picviz_seek_begin {
	seek($main::LOGFILE,0,0);
}

# Returns 0 if and only if we know for sure that this script can't decode the previous opened file
sub picviz_pre_discovery {
	return 1;
}

# Returns 1 if and only if text data are processed
sub picviz_is_element_text {
	return 1;
}

# This method must return a double-dimensional array of the processed data.
# Any rows that does not contain the good number of elements (according to the corresponding format)
# will be discarded.
sub picviz_get_next_chunk {
	my ($min_chunk_size) = $_[0];
	$row = 0;
	$nbytes = 0;

	my(@array) = ();

	while(<$main::LOGFILE>) {
	    chomp;
		$nbytes += length;

	    if (/(\w{3}\s+\w{3}\s+\d+\s+\d+:\d+:\d+\s+\d+).*Start MID \d+ ICID \d+/) { ($timestart) = ($1); }
	    if (/.*ICID \d+ From:\s+(.*)/) { ($from) = ($1); }
	    if (/.*MID \d+ ready (\d+) bytes.*/) {($bytes) = ($1);}
	    if (/.*DKIM: signing with (.*) - matches.*/) {($signwith) = ($1);}
	    if (/.*New SMTP DCID \d+ interface (\d+\.\d+\.\d+\.\d+) address (\d+\.\d+\.\d+\.\d+) port (.*)/) {($interface, $address, $port) = ($1,$2,$3);}
	    if (/.*MID \d+ Subject (.*)/) { ($subject) = ($1); }
	    if (/.*MID \d+ ICID \d+ .* To: (.*)/) {($to) = ($1);}
	    
	    if (/.*Message finished MID.*/ )  {
			$array[$row][0] = $timestart;
			$array[$row][1] = $from;
			$array[$row][2] = $bytes;
			$array[$row][3] = $address;
			$array[$row][4] = $subject;
			$array[$row][5] = $to;
			$row++;
	    }

		if ($nbytes >= $min_chunk_size) {
			last;
		}

		next;
	}

# We must reverse because perl stack will POP from the end
    return reverse @array;
}

# Close the previous opened file
sub picviz_close {
	close $main::LOGFILE
}
