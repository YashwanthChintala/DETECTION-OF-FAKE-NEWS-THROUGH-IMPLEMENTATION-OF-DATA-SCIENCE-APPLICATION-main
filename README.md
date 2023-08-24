# DETECTION-OF-FAKE-NEWS-THROUGH-IMPLEMENTATION-OF-DATA-SCIENCE-APPLICATION
clg mini project jntuh approved


# 1.	INTRODUCTION 

The advent of the World Wide Web and the rapid adoption of social media platforms (such as Facebook and Twitter) paved the way for information dissemination that has never been witnessed in the human history before. Besides other use cases, news outlets benefitted from the widespread use of social media platforms by providing updated news in near real time to its subscribers. The news media evolved from newspapers, tabloids, and magazines to a digital form such as online news platforms, blogs, social media feeds, and other digital media formats. It became easier for consumers to acquire the latest news at their fingertips. Facebook referrals account for 70% of traffic to news websites. These social media platforms in their current state are extremely powerful and useful for their ability to allow users to discuss and share ideas and debate over issues such as democracy, education, and health. However, such platforms are also used with a negative perspective by certain entities commonly for monetary gain and in other cases for creating biased opinions, manipulating mindsets, and spreading satire or absurdity. The phenomenon is commonly known as fake news.

There has been a rapid increase in the spread of fake news in the last decade, most prominently observed in the 2016 US elections. Such proliferation of sharing articles online that do not conform to facts has led to many problems not just limited to politics but covering various other domains such as sports, health, and also science. One such area affected by fake news is the financial markets, where a rumour can have disastrous consequences and may bring the market to a halt.

Our ability to take a decision relies mostly on the type of information we consume; our world view is shaped on the basis of information we digest. There is increasing evidence that consumers have reacted absurdly to news that later proved to be fake. One recent case is the spread of novel corona virus, where fake reports spread over the Internet about the origin, nature, and behaviour of the virus. The situation worsened as more people read about the fake contents online. Identifying such news online is a daunting task.

In this project, we propose a solution to the fake news detection problem using the implementation of data science application along with the use of NLP ensemble approach. Our study explores different textual properties that could be used to distinguish fake contents from real. By using those properties, we train a LSTM network using various ensemble methods that are not thoroughly explored in the current literature. The LSTM networks have proven to be useful in a wide variety of applications, as the learning models have the tendency to learn and remember processed for a longer period of time when compared to traditional neural networks. This helps is prediction of the next appropriate word in the sentence by remembering the first word for a longer period of time.
























#2.	LITERATURE SURVEY

2.1 TITLE- Fake News Detection
AUTHORS- Akshay Jain Amey Kasbe.
YEAR-2018
ABSTRACT- Information preciseness on Internet, especially on social media, is an increasingly important concern, but web-scale data hampers, ability to identify, evaluate and correct such data, or so called "fake news," present in these platforms. In this paper, we propose a method for "fake news" detection and ways to apply it on Facebook, one of the most popular online social media platforms. This method uses Naive Baves classification model to predict whether a post on Facebook will be labelled as REAL or FAKE. The results may be improved by applying several techniques that are discussed in the paper. Received results suggest, that fake news detection problem can be addressed with machine learning methods.


2.2 TITLE- A Tool for Fake News Detection
AUTHORS- Bashar Al Asaad, Madalina Erascu
YEAR-2016
ABSTRACT- The word post-truth was considered by Oxford Dictionaries Word of the Year
2016. The word is an adjective relating to or denoting circumstances in which objective facts are less influential in shaping public opinion than appeals to emotion and personal belief. This leads to misinformation and problems in society. Hence, it is important to make effort to detect these facts and prevent them from spreading. In this paper we propose machine learning techniques, in particular supervised learning, for fake news detection. More precisely, we used a dataset of fake and real news to train a machine learning model using Scikit-learn library in Python. We extracted features from the dataset using text representation models like Bag-of-Words, Term Frequency Inverse Document Frequency (TF-IDF) and Bi-gram frequency. We tested two classification approaches, namely probabilistic classification and linear classification on the title and the content, checking if it is clickbait/no clickbait, respectively fake/real. The outcome of our experiments was that the linear classification works the best with
the TF-IDF model in the process of content classification. The Bi-gram frequency model gave the lowest accuracy for title classification in comparison with Bag-of-Words and TF-IDF.


2.3 TITLE- Fake News Detection Using Content-Based Features and Machine
Learning
AUTHORS- Okuhle Nada, Bertram Haskins
YEAR-2020
ABSTRACT- The problem of fake news is a complex problem and is accompanied with social and economic ramifications. Targeted individuals and entities may lose trustworthiness, credibility and ultimately, suffer from reputation damages to their brand. Economically, an individual or brand may see fluctuations in revenue streams. In addition, the complex nature of the human language makes the problem of fake news a complex problem to solve for currently available computational remedies. The fight against the spread of fake news is a multi-disciplinary effort that will require research, collaboration and rapid development of tools and paradigms aimed at understanding and combating false information dissemination. This study explores fake news detection techniques using machine learning technology. Using a feature set which captures article structure, readability, and the similarity between the title and body, we show such features can deliver promising results. In the experiment, we select 6 machine learning algorithms, namely, AdaBoost as AB, Decision Tree as DT, K-Nearest Neighbour as KNN, Random Forest as RF, Support Vector Machine as SVM and XGBoost as XGB. To quantify a classifier's performance, we use the confusion matrix model and other performance metrics. Given the structure of the experiment, we show the Support Vector Machine classifier provided the best overall results.





2.4 TITLE- A Comprehensive Review on Fake News Detection with Deep
Learning
AUTHORS- M. F. Mridha; Ashfia Jannat Keya; Md. Abdul Hamid.
YEAR-2019
ABSTRACT- A protuberant issue of the present time is that, organizations from different domains are struggling to obtain effective solutions for detecting online-based fake news. It is quite thought-provoking to distinguish fake information on the internet as it is often written to deceive users. Compared with many machine learning techniques, deep learning-based techniques are capable of detecting fake news more accurately. Previous review papers were based on data mining and machine learning techniques, scarcely exploring the deep learning techniques for fake news detection. However, emerging deep learning-based approaches such as Attention, Generative Adversarial Networks, and Bidirectional Encoder Representations for Transformers are absent from previous surveys. This study attempts to investigate advanced and state-of-the-art fake news detection mechanisms pensively. We begin with highlighting the fake news consequences. Then, we proceed with the discussion on the dataset used in previous research and their NLP techniques. A comprehensive overview of deep learning-based techniques has been bestowed to organize representative methods into various categories. The prominent evaluation metrics in fake news detection are also discussed. Nevertheless, we suggest further recommendations to improve fake news detection mechanisms in future research directions.

2.5 TITLE- Automatic Online Fake News Detection Combining Content and Social Signals
AUTHORS- Marco L. Della Vedova; Eugenio Tacchini; Stefano Moret; Gabriele Ballarin
YEAR-2018
ABSTRACT-The proliferation and rapid diffusion of fake news on the Internet highlight the need of automatic hoax detection systems. In the context of social networks, machine learning (ML) methods can be used for this purpose. Fake news detection strategies are traditionally either based on content analysis (i.e. analysing the content of the news) or - more recently - on social context models, such as mapping the news' diffusion pattern. In this paper, we first propose a novel ML fake news detection method which, by combining news content and social context features, outperforms existing methods in the literature, increasing their already high accuracy by up to 4.8%. Second, we implement our method within a Facebook Messenger chatbot and validate it with a real-world application, obtaining a fake news detection accuracy of 81.7%.

2.6 SOFTWARE ENVIRONMENT
Python is a high-level, interpreted scripting language developed in the late 1980s by Guido van Rossum at the National Research Institute for Mathematics and Computer Science in the Netherlands. Python is an interpreted high-level programming language for general-purpose programming. Created by Guido van Rossum and first released in 1991, Python has a design philosophy that emphasizes code readability, notably using significant whitespace. 
Python features a dynamic type system and automatic memory management. It supports multiple programming paradigms, including object-oriented, imperative, functional and procedural, and has a large and comprehensive standard library. The initial version was published at the alt. Sources newsgroup in 1991, and version 1.0 was released in 1994.

Python 2.0 was released in 2000, and the 2. versions were the prevalent releases until December 2008. At that time, the development team made the decision to release version 3.0, which contained a few relatively small but significant changes that were not backward compatible with the 2x versions. Python 2 and 3 are very similar, and some features of Python 3 have been back ported to Python 2. But in general, they remain not quite compatible.

Both Python 2 and 3 have continued to be maintained and developed, with periodic release updates for both. As of this writing, the most recent versions available are 2.7.15 and 3.6.5.
However, an official End of Life date of January 1, 2020 has been established for Python 2, after which time it will no longer be maintained. If you are a newcomer to Python, it is recommended that you focus on Python 3, as this tutorial will do.

Python is still maintained by a core development team at the Institute, and Guido is still in charge, having been given the title of BDFL (Benevolent Dictator For Life) by the Python community. The name Python, by the way, derives not from the snake, but from the British comedy troupe Monty Python's Flying Circus, of which Guido was, and presumably still is, a fan. It is common to find references to Monty Python sketches and movies scattered throughout the Python documentation.


2.7 WHY CHOOSE PYTHON
If you're going to write programs, there are literally dozens of commonly used languages to choose from. Why choose Python? Here are some of the features that make Python an appealing choice.

Python is Popular

Python has been growing in popularity over the last few years. The 2018 Stack Overflow Developer Survey ranked Python as the 7th most popular and the number one most wanted technology of the year. World-class software development countries around the globe use Python every single day.

According to research by Dice Python is also one of the hottest skills to have and the most popular programming language in the world based on the Popularity of Programming Language
Index.

Due to the popularity and widespread use of Python as a programming language, Python developers are sought after and paid well. If you'd like to dig deeper into Python salary statistics and job opportunities, you can do so here.

Python is interpreted

Many languages are compiled, meaning the source code you create needs to be translated into machine code, the language of your computer's processor, before it can be run. Programs written in an interpreted language are passed straight to an interpreter that runs them directly.

This makes for a quicker development cycle because you just type in your code and run it, without the intermediate compilation step.

One potential downside to interpreted languages is execution speed. Programs that are compiled into the native language of the computer processor tend to run more quickly than interpreted programs. For some applications that are particularly computationally intensive, like graphics processing or intense number crunching, this can be limiting.

In practice, however, for most programs, the difference in execution speed is measured in milliseconds, or seconds at most, and not appreciably noticeable to a human user. The expediency of coding in an interpreted language is typically worth it for most applications.

Python is Free

The Python interpreter is developed under an OSI-approved open-source license, making it free to install, use, and distribute, even for commercial purposes.

A version of the interpreter is available for virtually any platform there is, including all flavors of Unix, Windows, macOS, smart phones and tablets, and probably anything else you ever heard of. A version even exists for the half dozen people remaining who use OS/2.

Python is Portable

Because Python code is interpreted and not compiled into native machine instructions, code written for one platform will work on any other platform that has the Python interpreter installed. (This is true of any interpreted language, not just Python.)

Python is Simple

As programming languages go, Python is relatively uncluttered, and the developers have deliberately kept it that way.

A rough estimate of the complexity of a language can be gleaned from the number of keywords or reserved words in the language. These are words that are reserved for special meaning by the compiler or interpreter because they designate specific built-in functionality of the language.
Python 3 has 33 keywords, and Python 2 has 31. By contrast, C+ + has 62, Java has 53, and Visual Basic has more than 120, though these latter examples probably vary somewhat by implementation or dialect.

Python code has a simple and clean structure that is easy to learn and easy to read. In fact, as you will see, the language definition enforces code structure that is easy to read.

But It's Not That Simple, for all its syntactical simplicity, Python supports most constructs that would be expected in a very high-level language, including complex dynamie data types, structured and functional programming, and object-oriented programming.

Additionally, a very extensive library of classes and functions is available that provides capability well beyond what is built into the language, such as database manipulation or GUI programming.

Python accomplishes what many programming languages don't: the language itself is simply designed, but it is very versatile in terms of what you can accomplish with it.
Tensorflow
TensorFlow is a free and open-source software library for dataflow and differentiable programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networks. It is used for both research and production at Google. 
TensorFlow was developed by the Google Brain team for internal Google use. It was released under the Apache 2.0 open-source license on November 9, 2015.
Numpy
Numpy is a general-purpose array-processing package. It provides a high-performance multidimensional array object, and tools for working with these arrays.
It is the fundamental package for scientific computing with Python. It contains various features including these important ones:
	A powerful N-dimensional array object
	Sophisticated (broadcasting) functions
	Tools for integrating C/C++ and Fortran code
	Useful linear algebra, Fourier transform, and random number capabilities
Besides its obvious scientific uses, Numpy can also be used as an efficient multi-dimensional container of generic data. Arbitrary data-types can be defined using Numpy which allows Numpy to seamlessly and speedily integrate with a wide variety of databases.
Pandas
Pandas is an open-source Python Library providing high-performance data manipulation and analysis tool using its powerful data structures. Python was majorly used for data munging and preparation. It had very little contribution towards data analysis. Pandas solved this problem. Using Pandas, we can accomplish five typical steps in the processing and analysis of data, regardless of the origin of data load, prepare, manipulate, model, and analyze. Python with Pandas is used in a wide range of fields including academic and commercial domains including finance, economics, Statistics, analytics, etc.
Matplotlib
Matplotlib is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms. Matplotlib can be used in Python scripts, the Python and IPython shells, the Jupyter Notebook, web application servers, and four graphical user interface toolkits.Matplotlib tries to make easy things easy and hard things possible. You can generate plots, histograms, power spectra, bar charts, error charts, scatter plots, etc., with just a few lines of code. For examples, see the sample plots and thumbnail gallery.
For simple plotting the pyplot module provides a MATLAB-like interface, particularly when combined with IPython. For the power user, you have full control of line styles, font properties, axes properties, etc, via an object oriented interface or via a set of functions familiar to MATLAB users.
Scikit – learn
Scikit-learn provides a range of supervised and unsupervised learning algorithms via a consistent interface in Python. It is licensed under a permissive simplified BSD license and is distributed under many Linux distributions, encouraging academic and commercial use. 
Conclusion

This section gave an overview of the Python programming language, including:

•	A brief history of the development of Python
•	Some reasons why you might select Python as your language of choice

Python is a great option, whether you are a beginning programmer looking to learn the basics, an experienced programmer designing a large application, or anywhere in between. The basics of Python are easily grasped, and yet its capabilities are vast. Proceed to the next section to learn how to acquire and install Python on your computer.

Python is an open source programming language that was made to be easy-to-read and powerful. A Dutch programmer named Guido van Rossum made Python in 1991. He named it after the television show Monty Python's Flying Circus. Many Python examples and tutorials include jokes from the show.

Python is an interpreted language. Interpreted languages do not need to be compiled to run. A program called an interpreter runs Python code on almost any kind of computer. This means that a programmer can change the code and quickly see the results. This also means Python is slower than a compiled language like C, because it is not running machine code directly.

Python is a good programming language for beginners. It is a high-level language, which means a programmer can focus on what to do instead of how to do it. Writing programs in Python takes less time than in some other languages.

Python drew inspiration from other programming languages like C, C+4, Java, Perl, and Lisp.
Python has a very easy-to-read syntax. Some of Python's syntax comes from C, because that is the language that Python was written in. But Python uses whitespace to delimit code: spaces or tabs are used to organize code into groups. This is different from C. In C, there is a semicolon at the end of each line and curly braces ({3) are used to group code. Using whitespace to delimit code makes Python a very easy-to-read language.

Python use

Python is used by hundreds of thousands of programmers and is used in many places. Sometimes only Python code is used for a program, but most of the time it is used to do simple jobs while another programming language is used to do more complicated tasks.

Its standard library is made up of many functions that come with Python when it is installed. On the Internet there are many other libraries available that make it possible for the Python language to do more things. These libraries make it a powerful language; it can do many different things.


Some things that Python is often used for are:

• Web development
• Scientific programming
• Desktop GUIs
• Network programming
• Game programming.


#3.	SYSTEM ANALYSIS

3.1 EXISTING SYSTEM:
The Fake news detection system plays a major role in eliminating the society of any possible fake or tampered news that might cause any disrupt in the normal functioning of the society and eliminates any possible unforeseen danger.

However, many approaches have been made in the past to solve the problem of fake news or tackle the fake news and many fake news detection systems have been implemented in the past.
One of those existing system have made use of the Support Vector Machine (SVM)
algorithm to detect and classify the news articles as either fake or real.
SVM is an algorithm to extract the binary class based on the data given to the model. In the existing model, the work is to classify the article in two categories either true or false. (SVM) is a supervised machine learning algorithm that can be used for both regression and classification purposes.
However, the existing system had some drawbacks, we will try to overcome these drawbacks in our proposed system and try to attain better system with better results.

Disadvantages:

• SVM algorithm is not suitable for large data sets.
• In cases where the number of features for each data point exceeds the number of training data samples, the SVM will underperform.

3.2 PROPOSED SYSTEM

Fake news has become a bountiful problem which does not cease to create disturbance among the majority and needs to be treated in some way for the very purpose, we are proposing a system which implements various algorithms i.e., Natural Language Processing (NLP) and Long short-term memory for the detection of Fake news.
 
In this project we are describing concept to detect fake news from social media or document corpus using Natural Language Processing and attribution supervised learning estimator. News documents or articles will be uploaded to application and then by using Natural Language Processing to extract quotes, verbs and name entity recognition (extracting organizations or person names) from documents to compute score, verbs, quotes and name entity also called as attribution. Using supervised learning estimator we will calculate score between sum of verbs, sum of name entity and sum of quotes divided by total sentence length. If score greater than 0 then news will be considered as REAL and if less than 0 then new will be consider as FAKE.

Advantages:

• Better efficiency.
• Memory for long time periods.
• Reduces the number of features to minimum level.
• LSTM can easily handle noise and continuous values
 
#4.	FEASIBILITY STUDY
The feasibility of the project is analyzed in this phase and business proposal is put forth with a very general plan for the project and some cost estimates. During system analysis the feasibility study of the proposed system is to be carried out. This is to ensure that the proposed system is not a burden to the company. For feasibility analysis, some understanding of the major requirements for the system is essential.
Three key considerations involved in the feasibility analysis are:
•	ECONOMICAL FEASIBILITY
•	TECHNICAL FEASIBILITY
•	SOCIAL FEASIBILITY

4.1 Economic feasibility
This study is carried out to check the economic impact that the system will have on the organization. The amount of fund that the company can pour into the research and development of the system is limited. The expenditures must be justified. Thus the developed system as well within the budget and this was achieved because most of the technologies used are freely available. Only the customized products had to be purchased.

4.2 Technical feasibility
This study is carried out to check the technical feasibility, that is, the technical requirements of the system. Any system developed must not have a high demand on the available technical resources. This will lead to high demands on the available technical resources. This will lead to high demands being placed on the client. The developed system must have a modest requirement, as only minimal or null changes are required for implementing this system.

4.3 Social feasibility
The aspect of study is to check the level of acceptance of the system by the user. This includes the process of training the user to use the system efficiently. The user must not feel threatened by the system, instead must accept it as a necessity. The level of acceptance by the users solely depends on the methods that are employed.
#5.	SYSTEM REQUIREMENTS

5.1 HARDWARE REQUIREMENTS:

•	System			:           Intel i3 5t gen or higher.
•	Hard Disk 		: 	512 GB or higher.
•	Input Devices		: 	Keyboard, Mouse
•	Ram			:	4 GB or higher
 

5.2 SOFTWARE REQUIREMENTS:

•	Operating system 		: 	Windows 7 or higher
•	Coding Language		:	python
•	Tool				:	IDLE Python 3.7 64bit)



















#6.	SYSTEM DESIGN

6.1 System Architecture:

 
Fig.6.1
An architecture diagram is a visual representation of all the elements that make a part, or on, of a system. Above all, it helps the engineers, designers, stakeholders and anyone else involved in the project understand the system or app's layout.

In the above architecture, the pre-processing part is being done with the help of the Natural Language Processing (NLP) toolkit, while the feature engineering part is done with the help of Long Short-term memory (LSTM) networks. The overall system is integrated together to help detect the news as fake or genuine.
 
6.2 DATA FLOW DIAGRAM:

1.	The DFD is also called as bubble chart. It is a simple graphical formalism that can be used to represent a system in terms of input data to the system, various processing carried out on this data, and the output data is generated by this system.
2.	The data flow diagram (DED) is one of the most important modeling tools. It is used to model the system components. These components are the system process, the data used by the process, an external entity that interacts with the system and the information flows in the system.
3.	DFD shows how the information moves through the system and how it is modified by a series of transformations. It is a graphical technique that depicts information flow and the transformations that are applied as data moves from input to output.
DFD is also known as bubble chart. A DFD may be used to represent a system at any level of abstraction. DD may be partitioned into levels that represent increasing information flow and functional detail.
 
Fig.6.2 
 
6.3 UML DIAGRAMS:
UML stands for Unified Modelling Language. UML is a standardized general-purpose modelling language in the field of object-oriented software engineering. The standard is managed, and was created by, the Object Management Group.
The goal is for UML to become a common language for creating models of object-oriented computer software. In its current form UML is comprised of two major components: a Meta-model and a notation. In the future, some form of method or process may also be added lo; or associated with, UML.
The Unified Modelling Language is a standard language for specifying, Visualization, Constructing and documenting the artifacts of software system, as well as for business modelling and other non-software systems.
The UML represents a collection of best engineering practices that have proven successful in the 
modelling of large and complex systems.
The UML is a very important part of developing objects-oriented software and the software development process. The UML uses mostly graphical notations to express the design of software projects.

GOALS:
     The Primary goals in the design of the UMI are as follows:
1.	Provide users a ready-to-use, expressive visual modelling Language so that they can develop and exchange meaningful models.
2.	Provide extendibility and specialization mechanisms to extend the core concepts.
3.	Be independent of particular programming languages and development process.
4.	Provide a formal basis for understanding the modelling language.
5.	Encourage the growth of 00 tools market.
6.	Support higher level development concepts such as collaborations, frameworks, patterns and components.
7.	Integrate best practices.
 
6.3.1 USE CASE DIAGRAM:
A use case diagram in the Unified Modelling Language (UML) is a type of behavioral diagram defined by and created from a Use-case analysis. Its purpose is to present a graphical overview of the functionality provided by a system in terms of actors, their goals (represented as use cases), and any dependencies between those use cases. The main purpose of a use case diagram is to show what system functions are performed for which actor. Roles of the actors in the system can be depicted.


 
Fig 6.3.1 
 
6.3.2 CLASS DIAGRAM:
In software engineering, a class diagram in the Unified Modelling Language (UMI) is a type
of static structure diagram that describes the structure of a system by showing the system's classes, their attributes, operations (or methods), and the relationships among the classes. It explains which class contains information.
 
Fig 6.3.2
 
6.3.3 SEQUENCE DIAGRAM:
A sequence diagram in unified modelling language (UML) is a kind of interaction diagram that shows how processes operate with one another and in what order. It is a construct of a message sequence chart. Sequence diagrams are sometimes called event diagrams, event scenarios, and timing diagrams.

 
Fig 6.3.3


















6.3.4 ACTIVITY DIAGRAM:

Activity diagrams are graphical representations of workflows of stepwise activities and
actions with support for choice, Iteration and concurrency. In the Unified Modelling Language, activity diagrams can be used to describe the business and operational step-by-step work lows of components in a system. an activity diagram shows the overall flow of control.





 
Fig 6.3.4 







6.3.5 COMPONENT DIAGRAM:
Component diagram is a special kind of diagram in UML. The purpose is also different from all other diagrams discussed so far. It does not describe the functionality of the system but it describes the components used to make those functionalities.
UML Component diagrams are used in modeling the physical aspects of object-oriented systems that are used for visualizing, specifying, and documenting component-based systems and also for constructing executable systems through forward and reverse engineering. Component diagrams are essentially class diagrams that focus on a system's components that often used to model the static implementation view of a system.

 
Fig 6.3.5 







#7. IMPLEMENTATION

7.1 MODULES:
	Data collection and uploading
	Data preprocessing
	Training the system
	Testing of data

MODULES DESCRIPTION:

1. Data collection and uploading
In this module the dataset containing various news articles is collected. This dataset is usually large and contain multiple fields which showcase the different news with their genuineness that is to be predicted
The collected dataset is then uploaded to our system which needs further pre-processing to be done in order for our system to be ready and running. The uploading of dataset might take some time since the dataset file is large in size.

2. Data pre-processing
The uploaded dataset is then ready for pre-processing. In this module the dataset uploaded is then pre-processed, i.e., the dataset is cleaned to remove defects such as anomalies, outliers, noise so that the dataset is ready for training and provides better Outcomes.
Since the uploaded dataset is usually it large it might take sometimes to pre-process yet the dataset works perfectly fine while running the algorithms.

3.Training the system
The system needs to be trained in order to be prepared to predict a facial expression. For this very reason we will be training the system using the dataset that we have uploaded and pre-processed.
 
4. Testing the data 
Testing the dataset can be another model where the system is ready to predict the news as either fake or genuine. In the module the system takes the test news dataset and predicts whether the news is fake or genuine. For this module to be ready all the previous modules have to be completed successfully. The testing data is taken from the dataset previously collected in the data collecting and uploading module. 

7.2 SAMPLE CODE-
```
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from sklearn.preprocessing import OneHotEncoder
import keras.layers
from keras.models import model_from_json
import pickle
import os
from sklearn.preprocessing import normalize

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM

main = Tk()
main.title("DETECTION OF FAKE NEWS THROUGH IMPLEMENTATION OF DATA SCIENCE APPLICATION")
main.geometry("1300x1200")

global filename
global X, Y
global tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test
global tfidf_vectorizer
global accuracy,error

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

textdata = []
labels = []
global classifier


def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def uploadDataset():    
    global filename
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="TwitterNewsData")
    textdata.clear()
    labels.clear()
    dataset = pd.read_csv(filename)
    dataset = dataset.fillna(' ')
    for i in range(len(dataset)):
        msg = dataset.get_value(i, 'text')
        label = dataset.get_value(i, 'target')
        msg = str(msg)
        msg = msg.strip().lower()
        labels.append(int(label))
        clean = cleanPost(msg)
        textdata.append(clean)
        text.insert(END,clean+" ==== "+str(label)+"\n")
    


def preprocess():
    text.delete('1.0', END)
    global X, Y
    global tfidf_vectorizer
    global tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test
    stopwords=stopwords = nltk.corpus.stopwords.words("english")
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf=True, ngram_range=(1,2),smooth_idf=False, norm=None, decode_error='replace', max_features=200)
    tfidf = tfidf_vectorizer.fit_transform(textdata).toarray()        
    df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names())
    text.insert(END,str(df))
    print(df.shape)
    df = df.values
    X = df[:, 0:df.shape[1]]
    X = normalize(X)
    Y = np.asarray(labels)
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = Y.reshape(-1, 1)
    print(X.shape)
    encoder = OneHotEncoder(sparse=False)
    #Y = encoder.fit_transform(Y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    print(Y)
    print(Y.shape)
    print(X.shape)
    tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"\n\nTotal News found in dataset : "+str(len(X))+"\n")
    text.insert(END,"Total records used to train machine learning algorithms : "+str(len(tfidf_X_train))+"\n")
    text.insert(END,"Total records used to test machine learning algorithms  : "+str(len(tfidf_X_test))+"\n")
    text.delete('1.0', END)
    global X, Y
    global tfidf_vectorizer
    global tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test
    stopwords=stopwords = nltk.corpus.stopwords.words("english")
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf=True, ngram_range=(1,2),smooth_idf=False, norm=None, decode_error='replace', max_features=200)
    tfidf = tfidf_vectorizer.fit_transform(textdata).toarray()        
    df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names())
    text.insert(END,str(df))
    print(df.shape)
    df = df.values
    X = df[:, 0:df.shape[1]]
    X = normalize(X)
    Y = np.asarray(labels)
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = Y.reshape(-1, 1)
    print(X.shape)
    encoder = OneHotEncoder(sparse=False)
    #Y = encoder.fit_transform(Y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    print(Y)
    print(Y.shape)
    print(X.shape)
    tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"\n\nTotal News found in dataset : "+str(len(X))+"\n")
    text.insert(END,"Total records used to train machine learning algorithms : "+str(len(tfidf_X_train))+"\n")
    text.insert(END,"Total records used to test machine learning algorithms  : "+str(len(tfidf_X_test))+"\n")


def runLSTM():
    text.delete('1.0', END)
    global classifier
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        classifier.load_weights("model/model_weights.h5")
        classifier._make_predict_function()   
        print(classifier.summary())
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        acc = acc[9] * 100
        text.insert(END,"LSTM Fake News Detection Accuracy : "+str(acc)+"\n\n")
        text.insert(END,'LSTM Model Summary can be seen in black console for layer details\n')
        with open('model/model.txt', 'rb') as file:
            classifier = pickle.load(file)
        file.close()
    else:
        lstm_model = Sequential()
        lstm_model.add(LSTM(128, input_shape=(X.shape[1:]), activation='relu', return_sequences=True))
        lstm_model.add(Dropout(0.2))

        lstm_model.add(LSTM(128, activation='relu'))
        lstm_model.add(Dropout(0.2))

        lstm_model.add(Dense(32, activation='relu'))
        lstm_model.add(Dropout(0.2))

        lstm_model.add(Dense(2, activation='softmax'))
        
        lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        hist = lstm_model.fit(X, Y, epochs=10, validation_data=(tfidf_X_test, tfidf_y_test))
        classifier = lstm_model
        classifier.save_weights('model/model_weights.h5')
        model_json = classifier.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        accuracy = hist.history
        f = open('model/history.pckl', 'wb')
        pickle.dump(accuracy, f)
        f.close()
        acc = accuracy['accuracy']                
        acc = acc[9] * 100
        text.insert(END,"LSTM Accuracy : "+str(acc)+"\n\n")
        text.insert(END,'LSTM Model Summary can be seen in black console for layer details\n')
        print(lstm_model.summary())
        
defget_image_path(image_lists, label_name, index, image_dir, category):
iflabel_name not in image_lists:
tf.logging.fatal('Label does not exist %s.', label_name)
label_lists = image_lists[label_name]
if category not in label_lists:
tf.logging.fatal('Category does not exist %s.', category)
category_list = label_lists[category]
if not category_list:
tf.logging.fatal('Label %s has no images in the category %s.',
label_name, category)
mod_index = index % len(category_list)
base_name = category_list[mod_index]
sub_dir = label_lists['dir']
full_path = os.path.join(image_dir, sub_dir, base_name)
returnfull_path


defget_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
category, architecture):
returnget_image_path(image_lists, label_name, index, bottleneck_dir,
category) + '_' + architecture + '.txt'
defcreate_model_graph(model_info):
withtf.Graph().as_default() as graph:
model_path = os.path.join(FLAGS.model_dir, model_info['model_file_name'])
withgfile.FastGFile(model_path, 'rb') as f:
graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())
bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
graph_def,
name='',
return_elements=[
model_info['bottleneck_tensor_name'],
model_info['resized_input_tensor_name'],
          ]))
return graph, bottleneck_tensor, resized_input_tensor
defrun_bottleneck_on_image(sess, image_data, image_data_tensor,
decoded_image_tensor, resized_input_tensor,
bottleneck_tensor):
resized_input_values = sess.run(decoded_image_tensor,
                                  {image_data_tensor: image_data})
bottleneck_values = sess.run(bottleneck_tensor,
                               {resized_input_tensor: resized_input_values})
bottleneck_values = np.squeeze(bottleneck_values)
returnbottleneck_values
defmaybe_download_and_extract(data_url):
dest_directory = FLAGS.model_dir
if not os.path.exists(dest_directory):
os.makedirs(dest_directory)
filename = data_url.split('/')[-1]
filepath = os.path.join(dest_directory, filename)
,
random_brightness):
return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
          (random_brightness != 0))
defadd_input_distortions(flip_left_right, random_crop, random_scale,
random_brightness, input_width, input_height,
input_depth, input_mean, input_std):
jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
margin_scale = 1.0 + (random_crop / 100.0)
resize_scale = 1.0 + (random_scale / 100.0)
margin_scale_value = tf.constant(margin_scale)
resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
minval=1.0,
maxval=resize_scale)
scale_value = tf.multiply(margin_scale_value, resize_scale_value)
precrop_width = tf.multiply(scale_value, input_width)
precrop_height = tf.multiply(scale_value, input_height)
precrop_shape = tf.stack([precrop_height, precrop_width])
precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
precropped_image = tf.image.resize_bilinear(decoded_image_4d,
precrop_shape_as_int)
  precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
cropped_image = tf.random_crop(precropped_image_3d,
                                 [input_height, input_width, input_depth])
ifflip_left_right:
flipped_image = tf.image.random_flip_left_right(cropped_image)
else:
flipped_image = cropped_image
brightness_min = 1.0 - (random_brightness / 100.0)
brightness_max = 1.0 + (random_brightness / 100.0)
brightness_value = tf.random_uniform(tensor_shape.scalar(),
minval=brightness_min,
maxval=brightness_max)
brightened_image = tf.multiply(flipped_image, brightness_value)
offset_image = tf.subtract(brightened_image, input_mean)
mul_image = tf.multiply(offset_image, 1.0 / input_std)
distort_result = tf.expand_dims(mul_image, 0, name='DistortResult')
returnjpeg_data, distort_result
defvariable_summaries(var):
withtf.name_scope('summaries'):
mean = tf.reduce_mean(var)
tf.summary.scalar('mean', mean)
withtf.name_scope('stddev'):
stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
tf.summary.scalar('stddev', stddev)
tf.summary.scalar('max', tf.reduce_max(var))
tf.summary.scalar('min', tf.reduce_min(var))
tf.summary.histogram('histogram', var)
defadd_final_training_ops(class_count, final_tensor_name, bottleneck_tensor,
bottleneck_tensor_size):
withtf.name_scope('input'):
bottleneck_input = tf.placeholder_with_default(

but found '%s' for architecture '%s'""",
size_string, architecture)
return None
iflen(parts) == 3:
is_quantized = False
else:
if parts[3] != 'quantized':
tf.logging.error(
            "Couldn't understand architecture suffix '%s' for '%s'", parts[3],
architecture)
return None
is_quantized = True
data_url = 'http://download.tensorflow.org/models/mobilenet_v1_'
data_url += version_string + '_' + size_string + '_frozen.tgz'
bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
bottleneck_tensor_size = 1001
input_width = int(size_string)
input_height = int(size_string)
input_depth = 3
resized_input_tensor_name = 'input:0'
ifis_quantized:
model_base_name = 'quantized_graph.pb'
else:
model_base_name = 'frozen_graph.pb'
model_dir_name = 'mobilenet_v1_' + version_string + '_' + size_string
model_file_name = os.path.join(model_dir_name, model_base_name)
input_mean = 127.5
input_std = 127.5
else:
tf.logging.error("Couldn't understand architecture name '%s'", architecture)
raiseValueError('Unknown architecture', architecture)

return {
      'data_url': data_url,
      'bottleneck_tensor_name': bottleneck_tensor_name,
      'bottleneck_tensor_size': bottleneck_tensor_size,
      'input_width': input_width,
      'input_height': input_height,
      'input_depth': input_depth,
      'resized_input_tensor_name': resized_input_tensor_name,
      'model_file_name': model_file_name,
      'input_mean': input_mean,
      'input_std': input_std,
  }


defadd_jpeg_decoding(input_width, input_height, input_depth, input_mean,
input_std):
jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
resize_shape = tf.stack([input_height, input_width])
resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
resized_image = tf.image.resize_bilinear(decoded_image_4d,
resize_shape_as_int)
offset_image = tf.subtract(resized_image, input_mean)
mul_image = tf.multiply(offset_image, 1.0 / input_std)
returnjpeg_data, mul_image

    
def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    loss = data['loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Epcchs')
    plt.ylabel('Accuracy/Loss')
    plt.plot(acc, 'ro-', color = 'green')
    plt.plot(loss, 'ro-', color = 'blue')
    plt.legend(['Accuracy','Loss'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title('LSTM Model Accuracy & Loss Graph')
    plt.show()

def predict():
    testfile = filedialog.askopenfilename(initialdir="TwitterNewsData")
    testData = pd.read_csv(testfile)
    text.delete('1.0', END)
    testData = testData.values
    testData = testData[:,0]
    print(testData)
    for i in range(len(testData)):
        msg = testData[i]
        msg1 = testData[i]
        print(msg)
        review = msg.lower()
        review = review.strip().lower()
        review = cleanPost(review)
        testReview = tfidf_vectorizer.transform([review]).toarray()
        predict = classifier.predict(testReview)
        print(predict)
        if predict == 0:
            text.insert(END,msg1+" === Given news predicted as GENUINE\n\n")
        else:
            text.insert(END,msg1+" == Given news predicted as FAKE\n\n")
        
    
font = ('times', 15, 'bold')
title = Label(main, text='DETECTION OF FAKE NEWS THROUGH IMPLEMENTATION OF DATA SCIENCE APPLICATION')
title.config(bg='gold2', fg='thistle1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Fake News Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=preprocess)
processButton.place(x=20,y=150)
processButton.config(font=ff)

dtButton = Button(main, text="Run LSTM Algorithm", command=runLSTM)
dtButton.place(x=20,y=200)
dtButton.config(font=ff)

graphButton = Button(main, text="Accuracy & Loss Graph", command=graph)
graphButton.place(x=20,y=250)
graphButton.config(font=ff)

predictButton = Button(main, text="Test News Detection", command=predict)
predictButton.place(x=20,y=300)
predictButton.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=330,y=100)
text.config(font=font1)

main.config(bg='DarkSlateGray1')
main.mainloop()
```



       # 8. SYSTEM TESTING

The purpose of testing is to discover errors. Testing is the process of trying to discover every conceivable fault or weakness in a work product. It provides a way to check the functionality of components, sub-assemblies, assemblies and/or a finished product It is the process of exercising software with the intent of ensuring that the Software system meets its requirements and user expectations and does not fail in an unacceptable manner. There are various types of test. Each test type addresses a specific testing requirement.

TYPES OF TESTS

Unit testing:
          Unit testing involves the design of test cases that validate that the internal program logic is functioning properly, and that program inputs produce valid outputs. All decision branches and internal code flow should be validated. It is the testing of individual software units of the application .it is done after the completion of an individual unit before integration. This is a structural testing, that relies on knowledge of its construction and is invasive. Unit tests perform basic tests at component level and test a specific business process, application, and/or system configuration. Unit tests ensure that each unique path of a business process performs accurately to the documented specifications and contains clearly defined inputs and expected results.

Integration testing:
Integration tests are designed to test integrated software components to determine if they actually run as one program.  Testing is event driven and is more concerned with the basic outcome of screens or fields. Integration tests demonstrate that although the components were individually satisfaction, as shown by successfully unit testing, the combination of components is correct and consistent. Integration testing is specifically aimed at   exposing the problems that arise from the combination of components.

Functional test:
Functional tests provide systematic demonstrations that functions tested are available as specified by the business and technical requirements, system documentation, and user manuals.
Functional testing is centered on the following items:

Valid Input:  identified classes of valid input must be accepted.
Invalid:  identified classes of invalid input must be rejected.
Functions:  identified functions must be exercised.
Output:  identified classes of application outputs must be exercised.
Systems:  interfacing systems or procedures must be invoked.

Organization and preparation of functional tests is focused on requirements, key functions, or special test cases. In addition, systematic coverage pertaining to identify Business process flows; data fields, predefined processes, and successive processes must be considered for testing. Before functional testing is complete, additional tests are identified and the effective value of current tests is determined.

System Test:
     System testing ensures that the entire integrated software system meets requirements. It tests a configuration to ensure known and predictable results. An example of system testing is the configuration-oriented system integration test. System testing is based on process descriptions and flows, emphasizing pre-driven process links and integration points.

White Box Testing:
        White Box Testing is a testing in which in which the software tester has knowledge of the inner workings, structure and language of the software, or at least its purpose. It is purpose. It is used to test areas that cannot be reached from a black box level.

Black Box Testing:
        Black Box Testing is testing the software without any knowledge of the inner workings, structure or language of the module being tested. Black box tests, as most other kinds of tests, must be written from a definitive source document, such as specification or requirements document, such as specification or requirements document. It is a testing in which the software under test is treated, as a black box. you cannot “see” into it. The test provides inputs and responds to outputs without considering how the software works.

8.1 Unit Testing:
Unit testing is usually conducted as part of a combined code and unit test phase of the software lifecycle, although it is not uncommon for coding and unit testing to be conducted as two distinct phases.
Test strategy and approach:
	Field testing will be performed manually and functional tests will be written in detail.

Test objectives:
•	All field entries must work properly.
•	Pages must be activated from the identified link.
•	The entry screen, messages and responses must not be delayed.

Features to be tested
•	Verify that the entries are of the correct format
•	No duplicate entries should be allowed
•	All links should take the user to the correct page.

8.2 Integration Testing

Software integration testing is the incremental integration testing of two or more integrated software components on a single platform to produce failures caused by interface defects.
	The task of the integration test is to check that components or software applications, e.g. components in a software system or – one step up – software applications at the company level – interact without error.
Test Results: All the test cases mentioned above passed successfully. No defects encountered.

8.3 Acceptance Testing

User Acceptance Testing is a critical phase of any project and requires significant participation by the end user. It also ensures that the system meets the functional requirements.
Test Results: All the test cases mentioned above passed successfully. No defects encountered.
























#9.	INPUT DESIGN AND OUTPUT DESIGN
9.1 INPUT DESIGN:
The input design is the link between the information system and the user. It comprises the developing specification and procedures for data preparation and those steps are necessary to put transaction data in to a usable form for processing can be achieved by inspecting the computer to read data from a written or printed document or it can occur by having people keying the data directly into the system. The design of input focuses on controlling the amount of input required, controlling the errors, avoiding delay, avoiding extra steps and keeping the process simple. The input is designed in such a way so that it provides security and ease of use with retaining the privacy. Input Design considered the following things:
	What data should be given as input?
	How the data should be arranged or coded?
	The dialog to guide the operating personnel in providing input.
	Methods for preparing input validations and steps to follow when error occur.

OBJECTIVES:
1. Input Design is the process of converting a user-oriented description of the input into a computer-based system. This design is important to avoid errors in the data input process and show the correct direction to the management for getting correct information from the computerized system.
2. It is achieved by creating user-friendly screens for the data entry to handle large volume of data. The goal of designing input is to make data entry easier and to be free from errors. The data entry screen is designed in such a way that all the data manipulates can be performed. It also provides record viewing facilities.
3. When the data is entered it will check for its validity. Data can be entered with the help of screens. Appropriate messages are provided as when needed so that the user will not be in maize of instant. Thus, the objective of input design is to create an input layout that is easy to follow



9.2 OUTPUT DESIGN:
A quality output is one, which meets the requirements of the end user and presents the information clearly. In any system results of processing are communicated to the users and to other system through outputs. In output design it is determined how the information is to be displaced for immediate need and also the hard copy output. It is the most important and direct source information to the user. Efficient and intelligent output design improves the system’s relationship to help user decision-making.
1. Designing computer output should proceed in an organized, well thought out manner; the right output must be developed while ensuring that each output element is designed so that people will find the system can use easily and effectively. When analysis design computer output, they should Identify the specific output that is needed to meet the requirements.
2. Select methods for presenting information.
3. Create document, report, or other formats that contain information produced by the system.
The output form of an information system should accomplish one or more of the following objectives.
	Convey information about past activities, current status or projections of the
	Future.
	Signal important events, opportunities, problems, or warnings.
	Trigger an action.
	Confirm an action.











#10.SCREENSHOTS
 
![image](https://user-images.githubusercontent.com/74952621/217644337-1d38508f-c8ca-4262-8314-dde492b88ece.png)

In above screen click on ‘Upload Fake News Dataset’ button to upload dataset








 
![image](https://user-images.githubusercontent.com/74952621/217644487-cfdbd847-e8b2-40fe-b292-a07127a2f88c.png)

In above screen selecting and uploading ‘news.csv’ file and then click on ‘Open’ button to load dataset and to get below screen










 
![image](https://user-images.githubusercontent.com/74952621/217644507-5937dbbb-107f-4a2f-a69a-c3b8bbc867c9.png)

In above screen dataset loaded and then in text area we can see all news text with the class label as 0 or 1 and now click on ‘Preprocess Dataset & Apply NGram’ button to convert above string data to numeric vector and to get below screen









 
![image](https://user-images.githubusercontent.com/74952621/217644571-6f7a745c-373e-4edc-91ac-71623b584af6.png)

In above screen we can see dataset contains total 7613 news articles and out of which we have used 6090 records or news articles to train the dataset while the rest of the records are saved or to be used for testing purpose once the system is ready







 
![image](https://user-images.githubusercontent.com/74952621/217644614-cd4be800-c66a-4c9c-ae72-6a5a1d92d90e.png)

In above screen after running the LSTM algorithm the system is ready to detect fake news and the accuracy achieved for the system is 69.49%.
We can even open the console to see the LSTM model summary.







 
![image](https://user-images.githubusercontent.com/74952621/217644643-53ef8804-bfbf-4aa7-af2e-5d8a062796ae.png)

In above screen different LSTM layers are created to filter input data to get efficient features for prediction. Now click on ‘Accuracy & Loss Graph’ button to get LSTM graph










 
![image](https://user-images.githubusercontent.com/74952621/217644707-9c39e966-5fcc-4b9e-bc62-7d5ffb439488.png)

Click on the accuracy and loss graph option to get the graph which shows the accuracy and the loss of the system.
In above screen x-axis represents epoch/iteration ad y-axis represents accuracy. Now click on
the test news detection option to detect the test news as either take or genuine.




 
![image](https://user-images.githubusercontent.com/74952621/217644745-ea77a1aa-64ea-47ff-b265-21fb07b4dacb.png)

In above screen in test news we have only one column which contains only news ‘TEXT’ and after applying above test news we will get prediction result










 
![image](https://user-images.githubusercontent.com/74952621/217644770-5b0b16a6-a41e-4b3f-933b-b6c11f302ddb.png)

In above screen selecting and uploading ‘testNews.txt’ file and then click on ‘Open’ button to load data and to get below prediction result







 
![image](https://user-images.githubusercontent.com/74952621/217644789-bb83ba4f-b561-4ff8-8cd1-5a5f9102bcbf.png)

After selecting and uploading the 'testnews.txt' file you will get the above screen which displays the result obtained after detection from the system.
In above screen all the test news articles are displayed along with a message which shows that the news is either detected as FAKE or GENUINE. For example, the article which quotes, "family members of osama bin laden have died in ad airplane accident how ironic???mmhmm gov" has been predicted as FAKE meaning that the news article is fake and is false.
So, this is the overall output of the project.











#10. CONCLUSION
In this project identifying misinformation is done in online social media platforms, because information is circulated easily across the online community by unsupported sources. To be able to automatically detect fake news and stop misinformation circulation, can be useful in analyzing wrongful deeds. This project presented the results of a study that produced a limited fake news detection system. The work presented herein is novel in this topic domain in that it demonstrates the results of a full-spectrum research project that started with qualitative observations and resulted in a working quantitative model. The work presented in this paper is also promising, because it demonstrates a relatively effective level of machine learning classification for large fake news documents with only one extraction feature. Finally, additional research and work to identify and build additional fake news classification grammars is ongoing and should yield a more refined classification scheme for both fake news and direct quotes.



















#11.  FUTURE ENHANCEMENT

The future scope can be to fetch the news being circulated from twitter automatically and detect whether the news being circulated is fake or genuine.
After the detection of news if it turns out to be fake then automatically report bomb the tweet spreading the misinformation among the people to avoid any mishap and problems.
The work presented in this paper is also promising, because it demonstrates a relatively effective level of machine learning classification for large take news documents with only one extraction feature. Finally, additional research and work to identify and build additional fake news classification grammars is ongoing and should yield a more refined classification scheme for both fake news and direct quotes




















#12.	BIBLIOGRAPHY

 [1] M. Balmas, "When Fake News Becomes Real: Combined Exposure to Multiple News Sources and Political Attitudes of Inefficacy, Alienation, and Cynicism," Communic. Res., vol.
41, по. 3, рр. 430 454, 2014.
[2] C. Silverman and J. Singer-Vine, "Most Americans Who See Fake News Believe It, New Survey Says," BuzzFeed News, 06-Dec-2016.
[3] P. R. Brewer, D. G. Young, and M. Morreale, "The Impact of Real News about *Fake News": Intertextual Processes and Political Satire," Int. J. Public Opin. Res., vol. 25, no. 3, 2013.
[4] D. Berkowitz and D. A. Schwartz, "Miley, CNN and The Onion," Journal. Pract., vol. 10, no. 1, pp. 1-17, Jan. 2016.
[5] C. Kang, "Fake News Onslaught Targets Pizzeria as Nest of Child-Trafficking," New York Times, 21-Nov-2016.
[6] C. Kang and A. Goldman, "In Washington Pizzeria Attack, Fake News Brought Real Guns," New York Times, 05-Dec-2016
