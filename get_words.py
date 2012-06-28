#!/usr/bin/python

#-------------------------------------------------------
import sys, os, urllib, use_psyco
import html_utils, tokenize
import re

import socket
socket.setdefaulttimeout(5)

try:
    set
except NameError:
    from sets import Set as set   # Python 2.3 fallback

set_SentenceTerminators = set(['.', '!', '?', ';', '<', '>', '|', '[', ']', '(', ')', '{', '}'])


#-------------------------------------------------------
def find_context(_lstHTML, _lstPhrase):

        setContext = set()

        iNeighbourhood = 10
        iMaxWordSeperation = 4

        iWordSeperation = 0
        iPhrase = 0
        lstHTMLIndex = []

        sPhraseWord = _lstPhrase [iPhrase]


        i = 0
        while i < len(_lstHTML):
                sHtmlWord = _lstHTML [i]

                if sHtmlWord == sPhraseWord:
                        iPhrase += 1
                        if iPhrase >= len(_lstPhrase):
                                iStart = lstHTMLIndex [0] - iNeighbourhood
                                if iStart < 0:
                                        iStart = 0
                                iEnd = lstHTMLIndex [-1] + iNeighbourhood
                                if iEnd >= len(_lstHTML):
                                        iEnd = len(_lstHTML) - 1

                                # add the extracted phrase to setContext ...
                                sText = ' '.join(_lstHTML [iStart : iEnd])
                                setContext.add(sText)
                                # continue searching for additional phrases

                                iWordSeperation = 0
                                iPhrase = 0
                                sPhraseWord = _lstPhrase [iPhrase]
                                lstHTMLIndex = []

                        sPhraseWord = _lstPhrase [iPhrase]
                        lstHTMLIndex.append(i)

                elif (sHtmlWord in set_SentenceTerminators) or \
                        (iWordSeperation > iMaxWordSeperation):
                        if len(lstHTMLIndex) > 0:
                                i = lstHTMLIndex [0] + 1
                        iWordSeperation = 0
                        iPhrase = 0
                        sPhraseWord = _lstPhrase [iPhrase]
                        lstHTMLIndex = []

                else:
                        iWordSeperation += 1

                i += 1

        return setContext

#-------------------------------------------------------


def find_paragraphs(cached, _file, _sURL,baseline=False):
    extra_linebreaks = re.compile('</?p[^>]*>|<br><br>', re.IGNORECASE)
    cached = extra_linebreaks.sub('\n\n',cached)
    spaces = re.compile('&nbsp;?', re.IGNORECASE)
    quotes = re.compile('&quot;?|&#34;?', re.IGNORECASE)
    newlines = re.compile('\n\n\n+')
    amps = re.compile('&amp;?|&#38;?', re.IGNORECASE)
    numbers = re.compile('\[\d+\]')
    tag = re.compile('<[^>]*>')
    whitespace = re.compile('\s+')
    beg_white = re.compile('^\s*')
    url = re.compile('\s(\S+\.\S+)\s')
    sent_delim = re.compile('(\.|!|\?)\s')
    word_delim = re.compile('\s+')
    non_english = re.compile('[^A-Za-z0-9\s]')
    cached = newlines.sub('\n\n',cached)
    cached = amps.sub('&',quotes.sub('"',spaces.sub(' ',numbers.sub(' ',cached))))
    lines = cached.splitlines()
    holdover = ''
    last1 = ''
    last2 = ''
    for l in lines:
        l = l.strip()
        l = tag.sub(' ', l)
        l = whitespace.sub(' ', l)
        l = url.sub(' ',l)
        l = beg_white.sub('', l)
        non = non_english.findall(l)
        if float(len(non)) > 0.1*float(len(l)):
            continue
        if l == '':
            holdover = tag.sub('',holdover)
            update = False
            if len(sent_delim.findall(holdover)) > 2 and len(sent_delim.findall(holdover)) < 15 and len(word_delim.findall(holdover)) > 15 and len(word_delim.findall(holdover)) < 400:
                print holdover + " " + _sURL
                print >> _file, holdover + " " + _sURL
                update = True
                if len(last1) > 0:
                    plus1 = last1+" "+holdover
                    if len(last2) > 0:
                        plus2 = last2+" "+last1+" "+holdover
                    else:
                        plus2 = ""
                else:
                    plus1 = ""
                    if len(last2) > 0:
                        plus2 = last2+" "+holdover
                    else:
                        plus2 = ""
                if baseline and len(plus1) > 2 and len(sent_delim.findall(plus1)) > 2 and len(sent_delim.findall(plus1)) < 15 and len(word_delim.findall(plus1)) > 15 and len(word_delim.findall(plus1)) < 400:
                    print plus1
                    print >> _file, plus1
                if baseline and len(plus2) > 2 and len(sent_delim.findall(plus2)) > 2 and len(sent_delim.findall(plus2)) < 15 and len(word_delim.findall(plus2)) > 15 and len(word_delim.findall(plus2)) < 400:
                    print plus2
                    print >> _file, plus2
            if update:
                last2 = last1
                last1 = holdover
            holdover = ''
        elif holdover == '':
            holdover = l
        else:
            holdover = holdover + ' ' + l


#-------------------------------------------------------
def process_search_result(_sURL, _sPhrase, _file, baseline=False):

        sHTML = ''
        # try using google cached link for now.  This will be faster,
        # hopefully.
        #cacheURL = 'http://www.google.com/search?q=cache:' + _sURL
        # looks like google cache is broken now.

        try:
                pURL = urllib.urlopen(_sURL)
                sHTML = pURL.read()
                pURL.close()
        except:
                #return []
                #print "Timeout!"
                return

        sys.stderr.write('\n'+_sURL+'\n')

        find_paragraphs(sHTML, _file,_sURL,baseline)

#        sHTML = sHTML.lower()
#        sHTML = sHTML.replace('\\"', '"')
#        sHTML = sHTML.replace('<b>', ' ')
#        sHTML = sHTML.replace('</b>', ' ')
#        sHTML = sHTML.replace('<i>', ' ')
#        sHTML = sHTML.replace('</i>', ' ')
#        sHTML = sHTML.replace('<ul>', ' ')
#        sHTML = sHTML.replace('</ul>', ' ')
#        sHTML = sHTML.replace('<p>', ' ')
#        sHTML = sHTML.replace('</p>', ' ')
#        sHTML = html_utils.preprocess(sHTML);

        # tokenize html before splitting !
#        sHTML = tokenize.tokenize_text(sHTML)
#        sPhrase = tokenize.tokenize_text(_sPhrase)
#        lstHTML = sHTML.split()
#        lstPhrase = sPhrase.split()

#        return find_context(lstHTML, lstPhrase)



#-------------------------------------------------------
def websearch_yahoo(_sPhrase, _iMaxResults, _iMaxContexts, _file,baseline=False):

        lstCleanLinks = []
        sURL = "http://search.yahoo.com/search?p=" + _sPhrase.replace(' ', '+')

        setContext = set()

        iPages = 0
#        while (iPages < _iMaxResults):
        pURL = urllib.urlopen(sURL)
        sHTML = pURL.read()
        pURL.close()

        iLinksStart = sHTML.find('<ol')
        iLinksEnd = sHTML.rfind('</ol>')

        lstLinks = html_utils.extract_tags(sHTML [iLinksStart : iLinksEnd], 'a')

        for sLink in lstLinks:
                (link, name) = html_utils.extract_link(sLink)
                name = html_utils.strip_tags(name)
                if 'Cached' != name:
                    lstCleanLinks.append((link, name))
                                #print '   ' + str(len(setContext)) + '   processing link : ' + name
                                #setNew = process_search_result(link, _sPhrase)
                    process_search_result(link, _sPhrase, _file, baseline)
#                                setContext.update(setNew)
#                                if len(setContext) >= _iMaxContexts:
#                                        return setContext
        
#                iNextStart = sHTML.rfind('<span>Next</span>')
#                iNextStart = sHTML.rfind('<a href', 0, iNextStart)
#                sLinkNext = html_utils.extract_tag(sHTML [iNextStart:], 'a', 0)
#               (link, name) = html_utils.extract_link(sLinkNext [0])

#                sURL = link
#                iPages += 1
                

#        return setContext

def retrieve_cache(_filename):
        
        infile = open(_filename,"r")
        
        for line in infile.readlines():
            print line,

        infile.close()


#-------------------------------------------------------
def main():

        name = sys.argv[1]
        name = name[0:name.find(' ')]
        filename = "data/" + name
        cat = sys.argv[1][sys.argv[1].find(' ')+1:]
        query = "\"" + name.lower().replace('_', ' ') + "\" " + cat.lower().replace('_', ' ');
        baseline = False
        if cat == "":
            baseline = True
        if not os.path.exists(filename):
            os.mkdir(filename)
        filename += "/" + cat.lower().replace(' ', '_')
        sys.stderr.write("Cache filename: " + filename)

        # try getting cache

        try:
            retrieve_cache(filename)
        except IOError:
            outfile = open(filename, "w")
            sSearchPhrase = query.replace('_', ' ')
            sys.stderr.write("Searching on: " + sSearchPhrase)
            #setContext = websearch_yahoo(sSearchPhrase, 1000, 20)
            websearch_yahoo(sSearchPhrase, 1000, 20, outfile,baseline)
            outfile.close()
        



        #file = open(sSearchPhrase.replace(' ', '_') + '.out', 'w')
        #for i, sContext in enumerate(setContext):
        #        file.write(str(i) + '. ' + sContext + '\n')
        #file.close()



if __name__ == '__main__':
        sys.exit(main())

