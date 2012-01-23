#!/usr/bin/perl

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
	print "From PERL: picviz_get_next_chunk $min_chunk_size\n";
	$row = 0;
	$nbytes = 0;

	my(@array) = ();

	while(<$main::LOGFILE>) {
	    # $line = $_;
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
#	print "$timestart,$from,$bytes,$signwith,$interface,$address,$port,$subject,$to\n";
		# print "\"$timestart\",\"$from\",\"$bytes\",\"$address\",\"$subject\",\"$to\"\n";
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

sub picviz_close {
	close $main::LOGFILE
}
