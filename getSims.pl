#!/usr/bin/perl

use strict;
use Text::Document;
use Text::DocumentCollection;
use porter;

$| = 1;

if (@ARGV != 1) {
    print "Usage: ./getSims.pl <paras>\n";
    exit(0);
}

my $para_file = $ARGV[0];

open STOPS, "english.stop" or die "Could not open english.stop";

my %stops;

for (my $line = <STOPS>) {
    chomp $line;
    $stops{$line} = 1;
}


open PARAS, $para_file or die "Could not open $para_file";

my $coll = Text::DocumentCollection->new(file => '.sims.db');
my $num = 0;
my @docs;

while (my $line = <PARAS>) {

    $line =~ s/[^A-Za-z0-9]/ /g;

    my @words = split(/\s+/, $line);
    foreach my $w (@words) {
        $w = lc($w);
        if ($stops{$w}) { $w = ""; }
        else { $w = porter($w); }
    }

    my $doc = Text::Document->new();
    $doc->AddContent(join(' ', @words));
    push @docs, $doc;
    $coll->Add($num, $doc);

    $num++;

}

close PARAS;

open SIMS, ">$para_file.sim" or die "Could not open $para_file.sim";
    
for (my $i = 0; $i < $#docs; $i++) {
    for (my $j = $i+1; $j < @docs; $j++) {
        my $cossim = $docs[$i]->CosineSimilarity($docs[$j]);
        print SIMS "$i $j $cossim\n";
    }
}

close SIMS;
