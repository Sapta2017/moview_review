import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# This is how the Naive Bayes classifier expects the input
def create_word_features(words):
    useful_words = [word for word in words if word not in stopwords.words("english") if word not in string.punctuation ]
    #useful_words = [word for word in words if word not in stopwords.words("english")]
    print useful_words
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict
    
neg_reviews = []
  
for fileid in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileid)
    neg_reviews.append((create_word_features(words), "negative"))
    #print(neg_reviews[0])
print(len(neg_reviews))

pos_reviews = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_reviews.append((create_word_features(words), "positive"))  
print(len(pos_reviews))


train_set = neg_reviews[:750] + pos_reviews[:750]
test_set =  neg_reviews[750:] + pos_reviews[750:]
print(len(train_set),  len(test_set))

classifier = NaiveBayesClassifier.train(train_set)
accuracy = nltk.classify.util.accuracy(classifier, test_set)

print(accuracy * 100)

review_BladeRunnr='''So what I can tell you about Blade Runner 2049 without having the spoiler police on my ass? Elvis is in it; ditto Sinatra. (I won't say how for fear of reprisals.) I can reveal that the sequel runs 2 hours and 43 minutes – that's 45 minutes longer than the 1982 original, based on Philip K. Dick's Do Androids Dream of Electric Sheep? For Blade Runner junkies like myself, who've mainlined five different versions of Ridley Scott's now iconic sci-fi film noir – from the release print to the Director's Cut and the Final Cut (the last two minus that voiceover Scott and Ford hated) – every minute of this mesmerizing mindbender is a visual feast to gorge on. 
Harrison Ford, at his hard-case best, comes roaring back as Rick Deckard, long past his days as blade runner in a relentlessly rainy Los Angeles circa 2019, when his job was to "retire" replicants. When four of these off-world androids escaped and wound up in the City of Angels, he was the assassin assigned to waste the quartet before they could pass as one of us. But then Deckard ran off with a repli-cutie named Rachael to do ... what? Marriage, kids, the whole nine yards? Is that even possible? The first film left us hanging. Scott went on record saying that Deckard was himself a replicant – an idea which horrified Ford who argued for his humanity.
Cut to 2049: Deckard is still missing 30 years later, and Officer K (a superb, soulful Ryan Gosling) is the young blade runner on his tail. His combative boss (Robin Wright) thinks the M.I.A. detective is the key to something that could "break the world." She's not kidding, and that prospect intensifies once Jared Leto shows up as Wallace, a replicant designer and demigod who can't keep up with the demand for his product. As for K's personal life, he lives with Joi (Ana de Armas), a virtual companion who can morph from homemaker to hottie in a nanosecond. But on those cold winter nights, how do you snuggle up to something transparent? Besides, K is haunted by something in his own past? An implanted memory or the real thing?
Am I being vague enough? I can reveal that French-Canadian director Denis Villeneuve (Arrival, Sicario), taking over for Scott – on board as an executive producer – shows a poet's eye for details that reveal emotion. The Blade Runner mythos could not be in better hands. And it's good to have the first film's screenwriter Hampton Fancher return, joined this time by Michael Green, to keep that philosophical Philip K. Dick spirit alive. And camera genius Roger Deakins (give this man an Oscar already!) creates a look that both salutes and moves on from the original, forging its own dazzling identity.
When K and Deckard finally meet – Gosling and Ford are double dynamite together – the film takes on a resonance that is both tragic and hopeful. It turns out that the theme of what it means to be human hasn't lost its punch, certainly not in a Trumpian era when demands are made on dreamers to prove their human worth. Blade Runner 2049, on its own march to screen legend, delivers answers – and just as many new questions meant to tantalize, provoke and keep us up nights. Would you have it any other way?'''

words = word_tokenize(review_BladeRunnr)
words = create_word_features(words)
classifier.classify(words)
