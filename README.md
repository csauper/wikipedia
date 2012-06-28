Automatically Generating Wikipedia Articles: A Structure-Aware Approach
===============================================================

Christina Sauper      | Regina Barzilay  
:--------------------:|:---------------------:
csauper@csail.mit.edu | regina@csail.mit.edu  

Abstract
--------

In this paper, we investigate an approach for creating a comprehensive textual
overview of a subject composed of information drawn from the Internet. We use
the high-level structure of human-authored texts to automatically induce a
domain-specific template for the topic structure of a new overview. The
algorithmic innovation of our work is a method to learn topicspecific
extractors for content selection jointly for the entire template. We augment
the standard perceptron algorithm with a global integer linear programming
formulation to optimize both local fit of information into each topic and
global coherence across the entire overview.  The results of our evaluation
confirm the benefits of incorporating structural information into the content
selection process.

Full text: http://people.csail.mit.edu/csauper/pubs/sauper-acl-09.pdf
Sample articles: http://people.csail.mit.edu/csauper/?page_id=64

Code
====

This code is available for research use only.

Running
-------

Run options are available by running `perceptron_ranker_full.py`.

Data format
-----------

A sample data file is provided in `data/sample.data`.  In general, for the
section /Section Name/ in the file `sections.section_name`, the format is as follows:

`##Article## !!section title!! body text ...`


