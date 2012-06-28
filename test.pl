#!/usr/bin/perl
#
# Outputs a string for use in maxent model.
#

use Math::Round;

use strict;
use porter;

if (@ARGV < 1 or @ARGV > 2) {
    print "Usage: features.pl <input> (<name>)\n";
    exit(0);
}

my $input_name = "";

if (@ARGV == 2) {
    $input_name = $ARGV[1];
}

my @name = split(/[_\s]+/, $input_name);
my %name_parts;

foreach my $n (@name) {
    if ($n =~ m/^[A-Z]/) {
        $name_parts{$n}++;
    }
}


my %stopwords = ("a", 1, "about", 1, "above", 1, "across", 1, "after", 1, "afterwards", 1, "again", 1, "against", 1, "all", 1, "almost", 1, "alone", 1, "along", 1, "already", 1, "also", 1, "although", 1, "always", 1, "am", 1, "among", 1, "amongst", 1, "amoungst", 1, "amount", 1, "an", 1, "and", 1, "another", 1, "any", 1, "anyhow", 1, "anyone", 1, "anything", 1, "anyway", 1, "anywhere", 1, "are", 1, "around", 1, "as", 1, "at", 1, "back", 1, "be", 1, "became", 1, "because", 1, "become", 1, "becomes", 1, "becoming", 1, "been", 1, "before", 1, "beforehand", 1, "behind", 1, "being", 1, "below", 1, "beside", 1, "besides", 1, "between", 1, "beyond", 1, "bill", 1, "both", 1, "bottom", 1, "but", 1, "by", 1, "call", 1, "can", 1, "cannot", 1, "cant", 1, "co", 1, "computer", 1, "con", 1, "could", 1, "couldnt", 1, "cry", 1, "de", 1, "describe", 1, "detail", 1, "do", 1, "done", 1, "down", 1, "due", 1, "during", 1, "each", 1, "eg", 1, "eight", 1, "either", 1, "eleven", 1, "else", 1, "elsewhere", 1, "empty", 1, "enough", 1, "etc", 1, "even", 1, "ever", 1, "every", 1, "everyone", 1, "everything", 1, "everywhere", 1, "except", 1, "few", 1, "fifteen", 1, "fify", 1, "fill", 1, "find", 1, "fire", 1, "first", 1, "five", 1, "for", 1, "former", 1, "formerly", 1, "forty", 1, "found", 1, "four", 1, "from", 1, "front", 1, "full", 1, "further", 1, "get", 1, "give", 1, "go", 1, "had", 1, "has", 1, "hasnt", 1, "have", 1, "he", 1, "hence", 1, "her", 1, "here", 1, "hereafter", 1, "hereby", 1, "herein", 1, "hereupon", 1, "hers", 1, "herself", 1, "him", 1, "himself", 1, "his", 1, "how", 1, "however", 1, "hundred", 1, "i", 1, "ie", 1, "if", 1, "in", 1, "inc", 1, "indeed", 1, "interest", 1, "into", 1, "is", 1, "it", 1, "its", 1, "itself", 1, "keep", 1, "last", 1, "latter", 1, "latterly", 1, "least", 1, "less", 1, "ltd", 1, "made", 1, "many", 1, "may", 1, "me", 1, "meanwhile", 1, "might", 1, "mill", 1, "mine", 1, "more", 1, "moreover", 1, "most", 1, "mostly", 1, "move", 1, "much", 1, "must", 1, "my", 1, "myself", 1, "name", 1, "namely", 1, "neither", 1, "never", 1, "nevertheless", 1, "next", 1, "nine", 1, "no", 1, "nobody", 1, "none", 1, "noone", 1, "nor", 1, "not", 1, "nothing", 1, "now", 1, "nowhere", 1, "of", 1, "off", 1, "often", 1, "on", 1, "once", 1, "one", 1, "only", 1, "onto", 1, "or", 1, "other", 1, "others", 1, "otherwise", 1, "our", 1, "ours", 1, "ourselves", 1, "out", 1, "over", 1, "own", 1, "part", 1, "per", 1, "perhaps", 1, "please", 1, "put", 1, "rather", 1, "re", 1, "same", 1, "see", 1, "seem", 1, "seemed", 1, "seeming", 1, "seems", 1, "serious", 1, "several", 1, "she", 1, "should", 1, "show", 1, "side", 1, "since", 1, "sincere", 1, "six", 1, "sixty", 1, "so", 1, "some", 1, "somehow", 1, "someone", 1, "something", 1, "sometime", 1, "sometimes", 1, "somewhere", 1, "still", 1, "such", 1, "system", 1, "take", 1, "ten", 1, "than", 1, "that", 1, "the", 1, "their", 1, "them", 1, "themselves", 1, "then", 1, "thence", 1, "there", 1, "thereafter", 1, "thereby", 1, "therefore", 1, "therein", 1, "thereupon", 1, "these", 1, "they", 1, "thick", 1, "thin", 1, "third", 1, "this", 1, "those", 1, "though", 1, "three", 1, "through", 1, "throughout", 1, "thru", 1, "thus", 1, "to", 1, "together", 1, "too", 1, "top", 1, "toward", 1, "towards", 1, "twelve", 1, "twenty", 1, "two", 1, "un", 1, "under", 1, "until", 1, "up", 1, "upon", 1, "us", 1, "very", 1, "via", 1, "was", 1, "we", 1, "well", 1, "were", 1, "what", 1, "whatever", 1, "when", 1, "whence", 1, "whenever", 1, "where", 1, "whereafter", 1, "whereas", 1, "whereby", 1, "wherein", 1, "whereupon", 1, "wherever", 1, "whether", 1, "which", 1, "while", 1, "whither", 1, "who", 1, "whoever", 1, "whole", 1, "whom", 1, "whose", 1, "why", 1, "will", 1, "with", 1, "within", 1, "without", 1, "would", 1, "yet", 1, "you", 1, "your", 1, "yours", 1, "yourself", 1, "yourselves", 1);
my %personals = ("i", 1, "me", 1, "my", 1, "you", 1, "your", 1, "we", 1, "us", 1, "our", 1);

open POS, $ARGV[0] or die "Could not open $ARGV[0]";

while (my $line = <POS>) {

    $line =~ s/!![^!]+!!//;
    $line =~ s/##[^#]+##//;
    
    my $excl = ($line =~ tr/!//);
    my $ques = ($line =~ tr/?//);
    my $sent = ($line =~ tr/.//) + $excl + $ques;
    
    $line =~ s/[^A-Za-z0-9]+/ /g;
    $line =~ s/^\s*//;
    $line =~ s/\s*$//;

    if ($line =~ m/^\s*$/) {
        next;
    }
    
    my @words = split(/\s+/,$line);
    my %uniq_words;
    my %bigrams;
    my %pos;
    my $pers = 0;

    my $prev = "***";
    my $i = 0;

    my %transl;

    foreach my $w (@words) {
        my $ew = $w;
        $w = porter($w);
        if ($w eq "") {
            next;
        }
        if ($personals{lc($w)} or $personals{lc($ew)}) {
            $pers++;
        }
        $transl{$w} = $ew;
        $uniq_words{$w}++;
        $bigrams{$prev . "@@" . $w}++;
        $prev = $w;
        push @{ $pos{$w} }, nearest(0.1,$i / @words);
        $i++;
    }
    $bigrams{$prev . "@@***"}++;

    print "TEST";

    my $dates = 0;
    my $nouns = 0;
    my $this_name = 0;

    my $first = uc($words[0]);
    my $second = uc($words[1]);

    my @sorted_words = sort {$a cmp $b} keys %uniq_words;

    foreach my $w (@sorted_words) {
        if ($w =~ m/^\s*$/) {
            next;
        }
        if ($w =~ m/^\d\d\d\d$/) {
            $dates += $uniq_words{$w};
            push @{ $pos{"DDDD"} }, @{ $pos{$w}};
            delete $pos{$w};
            if ($w eq $first) { $first = "DDDD";}
            if ($w eq $second) { $second = "DDDD"; }
            next;
        }
        if ($w =~ m/[A-Z]/) {
            if ($uniq_words{lc($w)}) {
                $uniq_words{lc($w)} += $uniq_words{$w};
            }
            elsif ($stopwords{lc($transl{$w})}) {
                print " " . uc($w) . ":$uniq_words{$w}";
            }
            elsif ($name_parts{$transl{$w}}) {
                $this_name += $uniq_words{$w};
                push @{ $pos{"SSSS"} }, @{ $pos{$w}};
                delete $pos{$w};
                if ($w eq $first) { $first = "SSSS"; }
                if ($w eq $second) { $second = "SSSS"; }
            }
            else {
                $nouns += $uniq_words{$w};
                push @{ $pos{"NNNN"} }, @{ $pos{$w}};
                delete $pos{$w};
            }
            next;
        }
        print " " . uc($w) . ":$uniq_words{$w}";
    }

    print " FIRST_$first";
    print " SECOND_$second";
    print " NNNN:$nouns";
    print " DDDD:$dates";
    if ($input_name ne "") {
        print " SSSS:$this_name";
    }
    print " LLLL:" . scalar(@words);
    print " QUES:$ques";
    print " EXCL:$excl";
    print " SENT:$sent";
    print " PERS:$pers";
    my $perp = $pers / scalar(@words);
    print " PERP:$perp";


    foreach my $b (keys %bigrams) {
        print " " . uc($b) . ":$bigrams{$b}";
    }

    foreach my $w (keys %pos) {
        my @sorted = sort {$a <=> $b} @{ $pos{$w} };
        print " #" . uc($w) . "#:$sorted[0]";
#        foreach my $wi (@sorted) {
#            print " #" . uc($w) . "#:$wi";
#        }
    }

    print "\n";
}

close POS;

