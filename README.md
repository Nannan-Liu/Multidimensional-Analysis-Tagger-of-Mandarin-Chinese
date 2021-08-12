<h1 id="multidimensional-analysis-tagger-of-mandarin-chinese">Multidimensional-Analysis-Tagger-of-Mandarin-Chinese</h1>
<p>MulDi Chinese (IPA: [ˌmʌl'daɪ] [ˌtʃaɪˈniːz]) is a multidimensional analysis tagger of Mandarin Chinese. 

 - Installation: `pip install muldichinese` 

<h2 id="About">About</h2>
Check the names of your input files, segment and pos tag the texts, and get the distribution of linguistic features and dimension scores of register variation

    from muldichinese.MulDiChinese import MulDiChinese
    mdc=MulDiChinese('/write/path/to/your/file(s)/')
    mdc.files()
    #print a list of your input files
    mdc.tag()
    #Segmentation and pos tagging completed.
    mdc.features()
    #Standardised frequencies of all 60 features written.
    mdc.dimensions()
    #Dimension scores written.

<h2 id="referencing-the-tagger">Reference the tagger</h2>
Liu, N. 2019. Multidimensional Analysis Tagger of Mandarin Chinese. Available at: <a href="https://github.com/Nannan-Liu/Multidimensional-Analysis-Tagger-of-Mandarin-Chinese">https://github.com/Nannan-Liu/Multidimensional-Analysis-Tagger-of-Mandarin-Chinese</a>.</p>
<p>This programme is based on the ICTCLAS, and it is advised to reference ICTCLAS when MulDi Chinese is used. Please refer to <a href="https://dl.acm.org/citation.cfm?id=1119280">https://dl.acm.org/citation.cfm?id=1119280</a>.</p>
<h2 id="requirements">Requirements</h2>
<p>Python packages needed are:

 1. <a href="https://pypi.org/project/PyNLPIR">PyNLPIR</a>
 2. <a href="https://www.nltk.org/#">NLTK</a>
 3. <a href="https://pandas.pydata.org/">Pandas</a>
 4. <a href="https://scikit-learn.org/stable/">scikit learn</a>
 5. <a href="https://numpy.org/">NumPy</a>

</p>
<h2 id="see-manual.pdf-for-more-details">See MulDi Chinese manual.pdf for more details</h2>
<p>The manual contains a detailed description of the 60 features.</p>
