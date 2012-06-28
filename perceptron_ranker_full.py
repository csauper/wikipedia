# Perceptron ranking implementation

from optparse import OptionParser
import numpy
from lpsolve55 import *
from lp_solve import *
from lp_maker import *
from nltk.tokenize import punkt
from nltk.corpus import gutenberg
import ds2
import time
import re
import porter
import os
import sys

DICT_FILENAME = "2of12inf.txt"
PRONOUN_FILENAME = "pronouns.txt"
STOP_FILENAME = "english.stop"

SECTIONS = ["diagnosis","causes","symptoms","treatment"]

BASE_DIR = "."

JOINT_LEARNING = True
MIN_RANK = True

currID = 0

class SectionCandidate(object):
    """ Represents a single section.  Contains a list of features. """

    def __init__(self, features, id=0, numSections=4, text="", url=""):
        """ Initializes with text and a list of features. """
        self.features = features
        self.ID = id
        self.indexes = [-1]*numSections
        self.text = text
        self.url=url

    def getFeatures(self):
        return self.features

    def getID(self):
        return self.ID

    def updateIndex(self, section, rank):
        self.indexes[section] = rank

    def getIndex(self):
        return self.indexes

    def getText(self):
        return self.text

    def getURL(self):
        return self.url

    def __eq__(self,other):
        return self.features == other.features

    def __str__(self):
        return self.text
            

class SectionCandidateList(object):
    """ Represents a list of candidates, including which is the best. """

    def __init__(self, candidates, best, sim):
        """ Initializes with a list of candidates and the best candidate
            for this section. """
        self.candidates = candidates
        self.best = best
        self.sims = sim
        self.scores = [0]*len(candidates)

    def getCandidates(self):
        return self.candidates

    def getNumCandidates(self):
        return len(self.candidates)

    def getBest(self):
        return self.best

    def getSims(self):
        return self.sims

    def sim(self,i,j):
        if i == j:
            return 1
        elif i < j:
            return self.sims[(i,j)]
        else:
            return self.sims[(j,i)]


    def sort(self, weights):
        """ Sorts candidates based on features*weights,
            from highest to lowest. """

        scores = {}

        for i in self.candidates:
            scores[i] = i.getFeatures() * weights


        # DSU optimized
        self.candidates[:] = [(scores[x], x) for x in self.candidates]
        self.candidates.sort()
        self.candidates.reverse()
        self.scores[:] = [key for (key, val) in self.candidates]
        self.candidates[:] = [val for (key, val) in self.candidates]

#        self.candidates.sort(key=lambda x: scores[x], reverse = True)
#        self.scores = sorted(scores.values(), reverse=True)

    def getScores(self):
        return self.scores
    
    def updateCandidateRanks(self, sectionNum):
        """ Updates indexes for each candidate """
        for i in range(len(self.candidates)):
            self.candidates[i].updateIndex(sectionNum, i)

    def __iter__(self):
        return self.candidates.__iter__()
    def __len__(self):
        return len(self.candidates)
    def __getitem__(self,index):
        return self.candidates[index]
                
    
class ArticleCandidateList(object):
    """ Represents several lists of candidates, including the best overall
        selection for the article. """
    
    def __init__(self, candidates):
        """ Initializes with a list of section candidate lists, as well as
            the best overall section selections. """
        self.candidates = candidates
        self.best = []
        self.numCands = 0
        for cands in self.candidates:
            if not cands is None:
                self.best.append(cands.getBest())
                self.numCands = len(cands)
            else:
                self.best.append(None)
        self.numSections = len(self.candidates)

    def getNumCandidates(self):
        return self.numCands

    def getNumSections(self):
        return self.numSections

    def getBest(self):
        return self.best

    def __iter__(self):
        return self.candidates.__iter__()
    def __getitem__(self,index):
        return self.candidates[index]

    def sortCandidateLists(self,weights):
        """ Sorts each candidate section based on weights """
        assert len(self.candidates) == len(weights), "Length mismatch"
        for i in range(len(self.candidates)):
            if not self.candidates[i] is None:
                self.candidates[i].sort(weights[i])
        for i in range(len(self.candidates)):
            if not self.candidates[i] is None:
                self.candidates[i].updateCandidateRanks(i)

    def get_top(self):
        """ Returns the best candidate for each section individually """
        best = []
        
        for cand in self.candidates:
            if not cand is None:
                best.append(cand[0])
            else:
                best.append(None)

        return best
            

    def do_lp(self):
        """ Performs ILP to find the joint best candidates """
#        print "\t\tFormulating LP"
        lp = self.formulate_lp()
#        print "\t\tSolving LP"
        sol = self.solve_lp(lp)
#        print "\t\tDone"
        return sol

    def formulate_lp(self):
        """ Formulates an ILP problem based on given cands and constrs. """
        A = []
        b = []
        e = []

        tmpSections = []
        for i in self.candidates:
            if not i is None:
                tmpSections.append(i)

        numCands = self.numCands
        numSections = len(tmpSections)
        totalElems = numSections*numCands

        # Template constraint
        constr = [0]*numCands*numSections

#        print "\t\t\tCreating objective"
        # Objective function: minimize sum of ranks
        if MIN_RANK:
            f = range(1,numCands+1)*numSections
        else:
            f = []
            for i in tmpSections:
                f += i.getScores()
        

#        print "\t\t\tOne candidate per section"
        # Constraint: Exactly one candidate must be chosen for each section
        for j in range(numSections):
            A.append([0]*numCands*j + [1]*numCands
                     + [0]*numCands*(numSections-j-1))
            b.append(1)
            e.append(0)

#        print "\t\t\tAll pairs similarity"
        # Constraint: (x1 + x2) * sim(x1,x2) <= 1
        
        for p1 in range(numCands - 1):
            for p2 in range(p1, numCands):
                sim = tmpSections[0].sim(tmpSections[0][p1].getID(), tmpSections[0][p2].getID())
                index1 = tmpSections[0][p1].getIndex()
                index2 = tmpSections[0][p2].getIndex()
                index1 = [x for x in index1 if x != -1]
                index2 = [x for x in index2 if x != -1]

                c = constr[:]
                for i in range(numSections):
                    if not tmpSections[i] is None:
                        c[i*numCands+index1[i]] = sim
                        c[i*numCands+index2[i]] = sim
                A.append(c)
                b.append(1.1)
                e.append(-1)

#        print "\t\t\tFormulate"
        
        # Minimize or maximize?
        obj = 0
        if MIN_RANK:
            obj = 1
        
        # Get LP formulation (but don't solve yet)
        lp = lp_maker(f,A,b,e,None,None,range(1,totalElems+1),1,obj)

        return lp

    def solve_lp(self,lp):
        """ Solve the lp formulation given.
            Return the candidates chosen.
            Delete LP when done. """
        
        sol = lpsolve('solve',lp)
        x = lpsolve('get_variables',lp)
        lpsolve('delete_lp',lp)

        chosen = []

        trueSections = 0

        for i in range(self.numSections):
            if self.candidates[i] is None:
                chosen.append(None)
                continue
            for j in range(self.numCands):
                # Float, so check for within epsilon of 1
                if abs(x[0][trueSections*self.numCands + j] - 1) < 0.0001:
                    chosen.append(self.candidates[i][j])
                    break
            trueSections += 1
                
        return chosen

class PerceptronILPRanker(object):
    """ Perceptron-ILP ranker.  Finds optimal solution based on
        perceptron rank updates and ILP redundancy detection. """

    def __init__(self, data, weights=None, sections=4):
        if weights is None:
            self.weights = []
            for i in range(sections):
                self.weights.append({})
        else:
            self.weights = weights  # Initial weights (probably 0s)
        self.data = data
        self.numArticles = len(self.data)

    def getWeights(self):
        return self.weights

    def train(self):
        """ Trains the perceptron. """
        if JOINT_LEARNING: 
            joint = "joint"
        else:
            joint = "sep"
        if MIN_RANK:
            rank = "rank"
        else:
            rank = "score"
    
        weightDir = BASE_DIR + "/full/%s-%s/" % (joint, rank)

        oldWeights = []
        counter = 0

        while oldWeights != self.weights and counter < 100:
            oldWeights = []
            for i in self.weights:
                oldWeights.append(i.copy())
            counter += 1

            print "Iteration %d" % counter,
            sys.stdout.flush()
            innercount = 0
            for acl in self.data:
                if innercount % (len(self.data) / 10) == 0:
                    print ".",
                    sys.stdout.flush()
#                print
#                for i in acl:
#                    if i is not None:
#                        print i[0].features.name
#                        break
#                print "\n\tGetting Best"
                best = acl.getBest()
                innercount += 1
                acl.sortCandidateLists(self.weights)
                if JOINT_LEARNING:
#                    print "\tDoing LP"
                    chosen = acl.do_lp()
                else:
#                    print "\tGetting Top"
                    chosen = acl.get_top()
                assert len(chosen) == len(best), "Length mismatch"
#                print "\tUpdating Features"
                for i in range(len(chosen)):
                    if chosen[i] != best[i]:
                        add = best[i].getFeatures().getFeatures()
                        sub = chosen[i].getFeatures().getFeatures()
                        for j in add:
                            if j in self.weights[i]:
                                self.weights[i][j] = self.weights[i][j] + add[j]
                            else:
                                self.weights[i][j] = add[j]
                        for j in sub:
                            if j in self.weights[i]:
                                self.weights[i][j] = self.weights[i][j] - sub[j]
                            else:
                                self.weights[i][j] = -sub[j]
            print
            saveWeights(weightDir,["diagnosis","causes","symptoms","treatment"],self.weights)
    

    def test(self):
        """ Tests perceptron based on prevous training. """
        results = []
        for acl in self.data:
            articleResult = []
            acl.sortCandidateLists(self.weights)

            if JOINT_LEARNING:
                chosen = acl.do_lp()
            else:
                chosen = acl.get_top()
            best = acl.getBest()
            assert len(chosen) == len(best), "Length mismatch"
            for i in range(len(chosen)):
                if best[i] is None:
                    continue
                else:
                    articleResult.append((chosen[i],best[i]))
            results.append(articleResult)

        return results



class FeatureSet(object):
    """ Set of features for a dictionary. """

    splitter = re.compile( "[a-z0-9]+(?:['\-][a-z0-9]+)*", re.I )
    dates = re.compile( r'\b\d\d\d\d\b|\'\d\d\b' )
    numbers = re.compile( "\d\+" )
    apos = re.compile("'$")
    stemmer = porter.PorterStemmer()
    words = None
    stop_words = None
    st = None
    wt = None
    pronouns = None
    fset = None
    
    def __init__(self, name="", features=None):
        """ Initializes a feature set. """

        # Load various libraries / dictionaries if they haven't been
        if FeatureSet.pronouns is None:
            FeatureSet.pronouns = loadDictionary(PRONOUN_FILENAME)
        if FeatureSet.words is None:
            FeatureSet.words = loadDictionary(DICT_FILENAME)
        if FeatureSet.stop_words is None:
            FeatureSet.stop_words = loadDictionary(STOP_FILENAME)
        if FeatureSet.st is None:
            # FeatureSet.st = punkt.PunktSentenceTokenizer(gutenberg.raw(gutenberg.files()))
            FeatureSet.st = punkt.PunktSentenceTokenizer()
        if FeatureSet.wt is None:
            FeatureSetwt = punkt.PunktWordTokenizer()

        # predefined set of features?
        if features is None:
            self.features = {}
        else:
            self.features = features
            
        # article name
        self.name = name

    def getFeatures(self):
        return self.features
    def getFeature(self, f):
        if f in self.features:
            return self.features[f]
        return 0

    def incrFeature(self,f):
        if f in self.features:
            self.features[f] += 1
        else:
            self.features[f] = 1
    def setFeature(self,f,val):
        self.features[f] = val

    def __iter__(self):
        return self.features.__iter__()
    def keys(self):
        return self.features.keys()
    def __eq__(self, other):
        for i in self.features:
            if i == "SIMS":
                continue
            if i in other.features:
                if other.features[i] != self.features[i]:
                    return False
            else:
                if self.features[i] != 0:
                    return False
        return True

    def __add__(self,other):
        f = self.features.copy()
        for i in other.features:
            if i in self.features:
                f[i] = self.features[i] + other.features[i]
            else:
                f[i] = other.features[i]
        return FeatureSet(f)

    def __radd__(self,other):
        for i in other.features:
            if i in self.features:
                self.features[i] += other.features[i]
            else:
                self.features[i] = other.features[i]

    def __mul__(self,other):
        dotprod = 0
        if type(other) == dict:
            fs = other
        else:
            fs = other.features
        for i in fs:
            if i in self.features:
                dotprod += self.features[i] * fs[i]

        return dotprod    
        
        
    def extractFeatures(self, text):
        """ Extracts features based on text.  Clears any existing features."""

        # Working text (will have things deleted)
        wtext = text

        # Clear dict before importing new features
        self.features = {}

        # Clean text
        words = FeatureSet.splitter.findall(text)

        # Number of words, sentences, questions, exclamations
        self.features["WORD"] = len(words)
        self.features["SENT"] = len(FeatureSet.st.tokenize(text))
        self.features["QUES"] = text.count("?")
        self.features["EXCL"] = text.count("!")

        # If we have an article name provided, find instances of that
        if (self.name != ""):
            occurs = 0
            namesplit = FeatureSet.splitter.findall(self.name)
            for i in namesplit:
                namepart = re.compile(r'\b'+i+r'\b',re.I)
                occurs += len(namepart.findall(text))
                wtext = namepart.sub("", wtext)
            self.features["NAME"] = occurs
            

        # Find dates
        self.features["DATE"] = len(FeatureSet.dates.findall(wtext))
        wtext = FeatureSet.dates.sub("", wtext)

        # Remove other numbers
        self.features["NUM"] = len(FeatureSet.numbers.findall(wtext))
        wtext = FeatureSet.numbers.sub("", wtext)

        # Now look for words / bigrams / positions
        pronouns = 0 # num pronouns
        propers = 0  # num proper nouns

        prev = "" # end marker
        i = -1.0
        length = len(wtext)
        wtext_words = FeatureSet.splitter.findall(wtext)
        for w in wtext_words:
            i += 1
            wl = w.lower()
            if wl in FeatureSet.pronouns:
                pronouns += 1
                continue
            if wl in FeatureSet.stop_words:
                # If this is a stop word, just ignore it
                continue
            if not wl in FeatureSet.words and wl != w:
                # Capital and not in word list, so assume it's a proper noun
                propers += 1
                continue

            ws = FeatureSet.stemmer.stem(wl,0,len(wl)-1)
            ws = FeatureSet.apos.sub("", ws)

            if FeatureSet.fset is None or "UNI_"+ws.upper() in FeatureSet.fset:
                self.incrFeature("UNI_"+ws.upper())
            if prev != "" and (FeatureSet.fset is None or "BI_"+prev.upper()+"_"+ws.upper() in FeatureSet.fset):
                self.incrFeature("BI_"+prev.upper()+"_"+ws.upper())
            if (not "POS_"+ws.upper() in self.features) and (FeatureSet.fset is None or "POS_"+ws.upper() in FeatureSet.fset):
                self.features["POS_"+ws.upper()] = i/length

            prev = ws

        firstword = FeatureSet.stemmer.stem(words[0],0,len(words[0])-1).upper()
        if FeatureSet.fset is None or "FIRST_"+firstword in FeatureSet.fset:
            self.features["FIRST_" + firstword] = 1
       
        if len(words) > 1:
            secondword = FeatureSet.stemmer.stem(words[1],0,len(words[1])-1).upper()
            if FeatureSet.fset is None or "SECOND_"+firstword+"_"+secondword in FeatureSet.fset:
                self.features["SECOND_" + firstword + "_" + secondword] = 1

        self.features["PROP"] = propers
        self.features["PRON"] = pronouns

def loadArticleList(datadir, article, sections, saveText=False, loadBest=True):
    """ Loads training examples from file given for one article. """

    global currID

    lparen = re.compile("\(")
    rparen = re.compile("\)")
    quote = re.compile("'")

    candidates = []
    best = []

    startID = currID
    
    # Get candidates for each paragraph
    filename = os.path.join(datadir, article,"all")
    secCands = loadCandidates(filename, article, saveText)
    simsfile = os.path.join(datadir,article,"all.sim")
    sims = loadSims(simsfile, startID)
    
    # similarity feature
    for i in range(len(secCands)-1):
        for j in range(i+1,len(secCands)):
            id1 = secCands[i].getID()
            id2 = secCands[j].getID()
            if id1 == id2:
                sim = 1
            elif id1 < id2:
                sim = sims[(id1,id2)]
            else:
                sim = sims[(id2,id1)]
            
            if sim > 0.5:
                secCands[i].getFeatures().incrFeature("SIMS")
                secCands[j].getFeatures().incrFeature("SIMS")
    
    candidates += secCands

    # Load best candidates (if exist), and append to overall candidate list
    for section in sections:
        filename = os.path.join(datadir, article, section+".best")
        b = loadCandidates(filename, article, saveText)
        if b == []:
            best.append(None)
        else:
            for i in secCands:
                if i == b[0]:
                    best.append(i)
                    break

    # Create candidate lists
    scls = []
    for i in range(len(sections)):
        if not loadBest:
            scls.append(SectionCandidateList(candidates[:],None,sims))
        elif best[i] is None:
            scls.append(None)
        else:
            scls.append(SectionCandidateList(candidates[:], best[i], sims))

    # Create article list
    data = ArticleCandidateList(scls)

    return data
    

def loadData(datadir, articles, sections,saveText=False):
    """ Loads training examples from file info given.

        datadir: base data directory where articles are stored (string)
        articles: names of articles to look at (list of strings)
        sections: names of sections to look at per article (list of strings)
        
    """
    global currID

    data = []

    lparen = re.compile("\(")
    rparen = re.compile("\)")
    quote = re.compile("'")

    # for each article
    for article in articles:
        print "\t%s" % article
        sys.stdout.flush()
        
        candidates = []
        best = []

#        article = lparen.sub("\\(", article)
#        article = rparen.sub("\\)", article)
 #       article = quote.sub("\\'", article)

        startID = currID
    
        # Get candidates for each paragraph
        filename = os.path.join(datadir, article,"all")
        secCands = loadCandidates(filename,article,saveText)
        simfilename = os.path.join(datadir,article,"all.sim")
        sims = loadSims(simfilename, startID)

        # similarity feature
        for i in range(len(secCands)-1):
            for j in range(i+1,len(secCands)):
                id1 = secCands[i].getID()
                id2 = secCands[j].getID()
                if id1 == id2:
                    sim = 1
                elif id1 < id2:
                    sim = sims[(id1,id2)]
                else:
                    sim = sims[(id2,id1)]
                
                if sim > 0.5:
                    secCands[i].getFeatures().incrFeature("SIMS")
                    secCands[j].getFeatures().incrFeature("SIMS")

        candidates += secCands

        # Load best candidates (if exist), and append to overall candidate list
        for section in sections:
            bestfile = os.path.join(datadir,article,section+".best")
            b = loadCandidates(bestfile,article,saveText)
            if b == []:
                best.append(None)
            else:
                thisBest = None
                for i in secCands:
                    if i == b[0]:
                        thisBest = i
                        break
                best.append(thisBest)

        # Create candidate lists
        scls = []
        for i in range(len(sections)):
            if best[i] is None and not saveText:
                scls.append(None)
            else:
                scls.append(SectionCandidateList(candidates[:], best[i], sims))

        # Create article list
        data.append(ArticleCandidateList(scls))

    return data

def loadSims(filename, startID = 0):
    """ Loads sims from given file """

    sims = {}
    try:
        f = open(filename)
        try:
            for line in f:
                i,j,sim = line.split()
                sims[(int(i)+startID,int(j)+startID)] = float(sim)
        finally:
            f.close()
    except IOError:
        pass

    return sims


def loadCandidates(filename, name="", saveText=False):
    """ Loads candidates from given file """
    global currID

    cands = []

    underbar = re.compile("_")
    name = underbar.sub(" ",name)
    
    try:
        f = open(filename)
        try:
            for line in f:
                line = line.strip()
                line,url = line.rsplit(" ", 1)
                fs = FeatureSet(name)
                fs.extractFeatures(line)
                if saveText:
                    cands.append(SectionCandidate(fs, currID,
                    text=line,url=url))
                else:
                    cands.append(SectionCandidate(fs, currID,url=url))
                currID += 1
        finally:
            f.close()
    except IOError:
        pass

    return cands

def loadDictionary(filename):
    """ Loads a dictionary of words for the purposes of detecting proper nouns. """
    s = set()
    f = open(filename)
    try:
        for line in f:
            line = line.strip()
            line = line.lower()
            s.add(line)
    finally:
        f.close()

    return frozenset(s)

def refreshOne(dataDir, article, sections, searchCommand, simCommand):
    for i in sections:
        os.system(searchCommand + ' "' + article + ' ' + i + '" &> /dev/null')
        
    try:
        filename = os.path.join(dataDir,article,'all')
        file = open(filename)
        file.close()
    except IOError:
        allLines = set()
#        article = lparen.sub("\\(", article)
#        article = rparen.sub("\\)", article)
#        article = quote.sub("\'", article)
        for i in sections:
            filename = os.path.join(dataDir,article,i)
            try:
                file = open(filename)
                try:
                    for line in file:
                        allLines.add(line)
                finally:
                    file.close()
            except IOError:
                pass
            filename = os.path.join(dataDir,article,i+".best")
            try:
                file = open(filename)
                try:
                    for line in file:
                        allLines.add(line+"\n")
                finally:
                    file.close()
            except IOError:
                pass
        filename = os.path.join(dataDir,article,'all')
        file = open(filename, 'w')
        
        try:
            for i in allLines:
                file.write(i)
        finally:
            file.close()

        filterResults(dataDir, article)

        os.system("rm .sims.db")
        os.system(simCommand + ' "' + filename + '"')

def refreshData(dataDir, bestsPrefix, sections, searchCommand, simCommand, namesOnly=False):
    """ Refreshes data directory, including the "best" examples.
        Returns a set of articles. """

    articles = set()
    name = re.compile("##([^#]+)##")
    body = re.compile("##[^#]+## !![^!]+!! (.*)")
    lparen = re.compile("\(")
    rparen = re.compile("\)")
    quote = re.compile("'")
    
    print "Refreshing text..."

    for i in sections:
        sectFile = open(bestsPrefix + i)
        try:
            for sect in sectFile:
                sect = sect.strip()
                article = name.match(sect).group(1)
                articles.add(article)
                if not namesOnly:
                    print "\t%s" % article
                    bodytext = body.match(sect).group(1)
#                    print searchCommand + ' "' + article + ' ' + i + '" &> /dev/null'
                    os.system(searchCommand + ' "' + article + ' ' + i + '" &> /dev/null')
                    filename = dataDir + "/" + article + "/" + i + ".best"
                    article = lparen.sub("\\(", article)
                    article = rparen.sub("\\)", article)
                    article = quote.sub("\\'", article)
                    try:
                        file = open(filename)
                        file.close()
                    except IOError:
                        outFile = open(filename, "w")
                        outFile.write(bodytext)
                        outFile.close()
        finally:
            sectFile.close()

    print "Refreshing similarities..."

    for article in articles:

        try:
            filename = os.path.join(dataDir,article,'all')
            file = open(filename)
            file.close()
        except IOError:
            print "\t%s" % article
            sys.stdout.flush()
            allLines = set()
#            article = lparen.sub("\\(", article)
#            article = rparen.sub("\\)", article)
#            article = quote.sub("\'", article)
            for i in sections:
                filename = os.path.join(dataDir,article,i)
                try:
                    file = open(filename)
                    try:
                        for line in file:
                            allLines.add(line)
                    finally:
                        file.close()
                except IOError:
                    pass
                filename = os.path.join(dataDir,article,i+".best")
                try:
                    file = open(filename)
                    try:
                        for line in file:
                            allLines.add(line+"\n")
                    finally:
                        file.close()
                except IOError:
                    pass
            filename = os.path.join(dataDir,article,'all')
            file = open(filename, 'w')
            
            try:
                for i in allLines:
                    file.write(i)
            finally:
                file.close()

            filterResults(dataDir, article)

            os.system("rm .sims.db")
            os.system(simCommand + ' "' + filename + '"')

    return frozenset(articles)

def filterResults(dataDir, article, featureCommand="./features.pl", filterCommand="maxent -m filter -o .result -p .features &> /dev/null"):
    filename = os.path.join(dataDir,article,"all")

    os.system(featureCommand + ' "' + filename + '" > .features')
    os.system(filterCommand)
    
    result = []

    try:
        file = open(".result")
        try:
            for line in file:
                line.strip()
                result.append(line)
        finally:
            file.close()
    except IOError:
        pass

    lines = []

    try:
        file = open(filename)
        try:
            i = 0
            for line in file:
                if result[i] != "not":
                    lines.append(line)
        finally:
            file.close()
    except IOError:
        pass

    os.system('rm "' + filename + '"')
    file = open(filename,"w")
    try:
        for i in lines:
            file.write(i)
    finally:
        file.close()

def saveWeights(dataDir, sections, weights):
    """ Save weights in files by section name """

    assert len(sections) == len(weights), "Length mismatch"

    for i in range(len(weights)):

        f = open(dataDir + "/" + sections[i] + ".weights", "w")
        
        for j in weights[i]:
            f.write(str(j) + "\t" + str(weights[i][j]) + "\n")
        f.close()

def loadWeights(dataDir, sections):
    """ Load weights from files by section name """

    weights = []

    for s in sections:
        w = {}
        f = open(dataDir + "/" + s + ".weights")
        try:
            for line in f:
                line = line.strip()
                feature, weight = line.split()
                w[feature] = float(weight)
        finally:
            f.close()
        weights.append(w)

    return weights
    
def evaluate(results, resultsFile):
    f = open(resultsFile, "w") 
    
    incorrect = 0
    total = 0
    per = []

    for result in results:
        article = 0
        for j in result:
            total += 1
            if not j[0] == j[1]:
                incorrect += 1
                article += 1
            f.write(str(j[0]) + "\n" + str(j[1]) + "\n\n")
        per.append(article / float(len(result)))
    
    total = float(total)

    print "\tPer article: ", per
    sys.stdout.flush()

    return (incorrect / total, sum(per) / float(len(results)), total, len(results))

def getArticle(article):
    sections = ["diagnosis","causes","symptoms","treatment"]
    dataDir = BASE_DIR + "/data"
    if JOINT_LEARNING: 
        joint = "joint"
    else:
        joint = "sep"
    if MIN_RANK:
        rank = "rank"
    else:
        rank = "score"

    weightDir = BASE_DIR + "/full/%s-%s/" % (joint, rank)

    searchCommand = BASE_DIR + "get_words.py"
    simCommand = BASE_DIR + "/getSims.pl"

#    print "Getting data..."
    sys.stdout.flush()
    refreshOne(dataDir, article, sections, searchCommand, simCommand)
    acl = loadArticleList(dataDir, article, sections, True, False)

#    print "Loading weights..."
    sys.stdout.flush()
    weights = loadWeights(weightDir,sections)
    
    acl.sortCandidateLists(weights)

    if JOINT_LEARNING:
        chosen = acl.do_lp()
    else:
        chosen = acl.get_top()

    for i in chosen:
        print str(i)

    urls = set()
    for i in chosen:
        urls.add(i.getURL())

    print "Sources:",len(urls)
    for i in chosen:
        print "\t",i.getURL()
    

                
def test():
    sections = SECTIONS
    dataDir = BASE_DIR + "data"
    if JOINT_LEARNING: 
        joint = "joint"
    else:
        joint = "sep"
    if MIN_RANK:
        rank = "rank"
    else:
        rank = "score"

    weightDir = BASE_DIR + "/full/%s-%s/" % (joint, rank)

    resultsFile = BASE_DIR + "/full/results"

    bestsPrefix = BASE_DIR + "/data/test/sections."
    searchCommand = BASE_DIR + "get_words.py"
    simCommand = BASE_DIR + "/getSims.pl"
    
    print "Refreshing data..."
    sys.stdout.flush()
    articles = refreshData(dataDir, bestsPrefix, sections, searchCommand, simCommand, True)
#    articles = refreshData(dataDir, bestsPrefix, sections, searchCommand, simCommand)

    print "Loading data..."
    sys.stdout.flush()
    data = loadData(dataDir,articles,sections,True)

    print "Loading weights..."
    sys.stdout.flush()
    weights = loadWeights(weightDir,sections)

    print "Creating ranker..."
    sys.stdout.flush()
    ranker = PerceptronILPRanker(data, weights)

    print "Testing ranker..."
    sys.stdout.flush()
    results = ranker.test()

    print "Evaluating results..."
    sys.stdout.flush()
    overall,per,total,numarts = evaluate(results,resultsFile)
    print "Out of %d sections in %d articles..." % (int(total),numarts)
    print "Overall accuracy: %0.2f" % (1-overall)
    print "Average number of sections correct: %0.2f" % (1-per)
    sys.stdout.flush()


def train(resume=False):

    sections = SECTIONS
    
    if JOINT_LEARNING: 
        print "Learning: JOINT"
        joint = "joint"
    else:
        print "Learning: INDIVIDUAL"
        joint = "sep"
    if MIN_RANK:
        print "Optimizing: RANK"
        rank = "rank"
    else:
        print "Optimizing: SCORE"
        rank = "score"

    weightDir = BASE_DIR + "/full/%s-%s/" % (joint, rank)

    dataDir = BASE_DIR + "data"

    bestsPrefix = BASE_DIR + "/data/train/sections."
    searchCommand = BASE_DIR + "get_words.py"
    simCommand = BASE_DIR + "/getSims.pl"

    print "Refreshing data..."
    sys.stdout.flush()
    articles = refreshData(dataDir, bestsPrefix, sections, searchCommand, simCommand, True)
#    articles = refreshData(dataDir, bestsPrefix, sections, searchCommand, simCommand)

    print "Loading data..."
    sys.stdout.flush()
    data = loadData(dataDir, articles, sections)

    if resume:
        print "Loading weights..."
        sys.stdout.flush()
        weights = loadWeights(weightDir,sections)

    print "Creating ranker..."
    sys.stdout.flush()
    if resume:
        ranker = PerceptronILPRanker(data,weights)
    else:
        ranker = PerceptronILPRanker(data)

    print "Training ranker..."
    sys.stdout.flush()
    ranker.train()

    print "Saving weights..."
    sys.stdout.flush()
    weights = ranker.getWeights()
    
    saveWeights(weightDir, sections, weights)
    

def test_features():
    fs = FeatureSet()
    text = "I was born in '84, and then I went to the moon.  Then, in 1985, I came back home a new person.  Can't you see that?"
    fs.extractFeatures(text)
    return text, fs.features

def test_formulate():
    s1 = SectionCandidate([],"This here be section one, mon!")
    s2 = SectionCandidate([],"Section two is so cool, man!")
    s3 = SectionCandidate([],"This here be section three, mon!")
    s4 = SectionCandidate([],"Did you know section four is really neat?")
    cands = [s1,s2,s3,s4]
    sl1 = SectionCandidateList(cands,s1)
    sl2 = SectionCandidateList(cands,s2)
    sl3 = SectionCandidateList(cands,s4)
    sl1.getSims()
    clists = [sl1,sl2,sl3]
    acl=ArticleCandidateList(clists)
    lp = acl.formulate_lp()

    sol = lpsolve('solve',lp)
    obj = lpsolve('get_objective',lp)
    x = lpsolve('get_variables',lp)
    lpsolve('delete_lp',lp)
    return obj, x

def test_solve():
    s1 = SectionCandidate([],"This here be section one, mon!")
    s2 = SectionCandidate([],"Section two is so cool, man!")
    s3 = SectionCandidate([],"This here be section three, mon!")
    s4 = SectionCandidate([],"Did you know section four is really neat?")
    cands = [s1,s2,s3,s4]
    sl1 = SectionCandidateList(cands,s1)
    sl2 = SectionCandidateList(cands,s2)
    sl3 = SectionCandidateList(cands,s4)
    sl1.getSims()
    clists = [sl1,sl2,sl3]
    acl=ArticleCandidateList(clists)
    lp = acl.formulate_lp()

    x = acl.solve_lp(lp)

    for i in x:
        print i.getText()

    print x == [s1,s3,s4]
        
    return x


def loadFeatures(fset):
    """ Loads applicable features from file given in fset """

    features = set()

    try:
        f = open(fset)
        try:
            for line in f:
                line = line.strip()
                features.add(line)
        finally:
            f.close()
    except IOError:
        pass

    return features

def main():
    usage = '%prog [options] command\n\n'
    usage += 'commands:  train  test  run  resume'
    parser = OptionParser(usage)
    
    parser.set_defaults(joint=True, rank=True, fset=None)

    parser.add_option("-j", "--joint", action="store_true", dest="joint",
                      help="Learn weightings jointly [default]")
    parser.add_option("-i", "--individual", action="store_false", dest="joint",
                      help="Learn weightings individually")
    parser.add_option("-r", "--min-rank", action="store_true", dest="rank",
                      help="Minimize rank during optimization [default]")
    parser.add_option("-s", "--max-score", action="store_false", dest="rank",
                      help="Maximize score during optimization")
    parser.add_option("-t", "--title", action="store", dest="title",
                      help="Title of article to generate (required for 'run')")
    parser.add_option("-f", "--features", action="store", dest="fset",
                      help="File containing usable feature set.")

    (options, args) = parser.parse_args()
    if len(args) < 1:
        parser.error("Please input a command")

    global JOINT_LEARNING
    JOINT_LEARNING = options.joint
    global MIN_RANK
    MIN_RANK = options.rank

    if not options.fset is None:
        FeatureSet.fset = loadFeatures(options.fset)

    command = args[0]
    
    if command == "run" and options.title == "":
        parser.error("The command \"run\" requires an article title")

    if command == "train":
        train()
    elif command == "test":
        test()
    elif command == "run":
        getArticle(args[1])
    elif command == "resume":
        train(True)
    else:
        parser.error("Incorrect command supplied")

   

if __name__ == "__main__":

    main()
