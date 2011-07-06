#!/usr/bin/perl

#use strict vars;

#our ($timestart,$from,$bytes,$signwith,$interface,$address,$port,$subject,$to);

sub picviz_format() {
    return " \
key-axes = \"2,9\" \
\
#Wed Jun  2 01:32:52 2010 \
#time-format[1] = \"\%a \%b  \%d \%H:\%M:\%S \%Y\" \
\
axes { \
   enum default default \"Time\" \
   enum default default \"From\" \
   integer default minmax \"Size\" \
   ipv4 default default \"Address\" \
   string default default \"Subject\" \
   enum default default \"To\" \
}\n";
}

sub picviz_normalize() {
    $filename = $_[0];

    $row = 0;
    open(IRONFILE, '<', $filename) or die $!;
    while(<IRONFILE>) {
	# $line = $_;
	chomp;

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
	    next;
	}

    }
    close IRONFILE;
    

# We must reverse because perl stack will POP from the end
    return reverse @array;
}


