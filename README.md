# DIY Projects
This repository contains few randomly undertaken projects in **Data Analytics** and **Machine Learning** domain. Intent is to explore few under-valued pre-created Python package/libraries as well as methodologies that can actually create a business impact. Mostly projects vary in content depth and if relevant for reference, each project is also associated with a descriptive run-through article on my [Medium](https://medium.com/@neuralnets) blog. Down the lane, also urge to showcase this repo to **Data Science** aspirants for helping them envision what they can undertake as projects to display their skills. 

If you're a learner, this repository can be your guide in building a portfolio for yourself where you get to practice on various **Machine Learning** concepts, along with **Deep Learning** applications. Feel free to **Star** for reference or **Fork** if willing to contribute. All of us need to take responsibility to add more to our open-source community, whichever way we can. :)

# Index:
* **WiFi spots Map of Networks** around us with [WiGLE](https://wigle.net/) and Python. Selected geographical co-ordinates were of *[Pune (India)](https://www.pmc.gov.in/en/geographic-information-systems)*. Static visual demonstrated using [Geoplotlib](https://pypi.org/project/geoplotlib/) and interactivity enhancement using [Folium](https://pypi.org/project/folium/). Static Output as well as [Interactive inference](https://www.youtube.com/watch?v=s1ACNtf6oS0) available for reference, along with a [step-by-step guide on my blog](https://medium.com/@neuralnets/building-a-wifi-spots-map-of-networks-around-you-with-wigle-and-python-5adf72a48140) for learners.

* **OCR (Optical Character Recognition) to parse files** containing complex table structured invoice screenshots for text recognition & extraction by applying Deep Learning, using [pytesseract v0.2.6](https://pypi.org/project/pytesseract/), [Pillow v5.4.1](https://pypi.org/project/Pillow/) and [OpenCV-4](https://pypi.org/project/opencv-python/). We feed forward images to our [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) [(RNN)](https://en.wikipedia.org/wiki/Recurrent_neural_network) network for text recognition. For testing purpose, we experiment on Walmart receipts as well as transaction tables.

* **Exploratory Frequency Analysis on NLP Data** of [IMDB movie reviews](https://www.kaggle.com/utathya/imdb-review-dataset) to gain an insight on buzz words amidst viewer comments. Acts as a pillar for further algorithmic processing! Natural language processing has been carried out using [NLTK v3.4](https://pypi.org/project/nltk/) and visual representation for word tagging is implemented using [Word Cloud](https://pypi.org/project/wordcloud/).
 
* **Coloring Black & White Images** by applying [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network). Selected architecture refers to Zhang et al.’s ECCV paper, [Colorful Image Colorization](http://richzhang.github.io/colorization/) where he trains [Imagenet](http://image-net.org/) dataset to [Lab](https://en.wikipedia.org/wiki/CIELAB_color_space) from RGV with mean annealing. Implementation is done using [OpenCV-4](https://pypi.org/project/opencv-python/), by utilizing pre-trained `.caffemodel` which contains *weights* for actual layers & `.prototxt` file which shall define our *model architecture*.

* **Manipulation of PDF Files** using core Python for varied use cases. Involves splitting a multi-page PDF file into individual page files, merging individual files into a single PDF file, slicing out selective pages from a multi-page PDF file based on index page numbers, and rotating all pages of a PDF file in either clockwise or anti-clockwise direction as per use. Involves usage of built-in `os` and `glob` packages, along with [PyPDF2](https://pypi.org/project/PyPDF2/) package.

* **Face Detection in static Images** by applying `face detector` capability of *OpenCV-4* `dnn` module, using pre-trained *Caffe Deep Learning* model based on the **Single Shot Detector (SSD)** framework with a [ResNet network](https://en.wikipedia.org/wiki/ResNet). Architecture includes `.prototxt` files from `samples/dnn/face_detector/` directory of [OpenCV-4](https://pypi.org/project/opencv-python/) GitHub repository & associated Weights for actual layers.

* **Marketing Campaign Subscription Analysis & Classification** by applying [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression) with k-Fold cross-validation technique to identify potential customers of a Portuguese bank who would subscribe to a *Term Deposit* using conventional [Scikit-Learn](https://scikit-learn.org/stable/) tools. Dataset utilized for this assignment is available at [UCI repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing). Few additional Python external libraries have also been used for EDA like [Seaborn](https://pypi.org/project/seaborn/) and [Yellowbrick](https://pypi.org/project/yellowbrick/).

* **Fake News detection with NLP techniques** by exploring dataset scraped from various websites during *US Presidential elections of  October 2016* and categorized per *biases* accordingly. Currently access is limited to just [Fake news dataset](https://www.kaggle.com/mrisdal/fake-news), and not on supplementing *Real news* dataset, hence attached worksheet displays exploratory data cleansing, preprocessing and [bigram](https://en.wikipedia.org/wiki/Bigram) analogy using [NLP](https://en.wikipedia.org/wiki/Natural_language_processing) tools like [NLTK](https://pypi.org/project/nltk/) and [Wordcloud](https://pypi.org/project/wordcloud/). With access to other set of data, we shall enhance project with tools like [SpaCy v2.0.18](https://pypi.org/project/spacy/) and [Gensim v3.7](https://pypi.org/project/gensim/) to perform *Deep Learning* for classifying the flavour of news source and content.

* **Traversing Airbnb London Calendar & Listings** as of February 5th, 2019 on unofficial [Inside Data](http://insideairbnb.com/get-the-data.html) application for hostings using extensive analysis, segmentation and visualization of various aspects. Further data being modelled and summarized with conventional [Scikit-Learn](https://scikit-learn.org/stable/), [LightGBM v2.2.3](https://pypi.org/project/lightgbm/) and [Keras v2.2.4](https://pypi.org/project/Keras/) deep neural network architecture for regression.

* **Segmentation and Object Recognition** using [Holistically-Nested Edge Detection](https://arxiv.org/abs/1504.06375) for obtaining this state-of-the-art computer vision technique. Deployment is based on the algorithm introduced in [Xie and Tu’s HED research](https://arxiv.org/abs/1504.06375) paper, and utilizes a pre-trained `.caffemodel` Caffemodel for *weights* along with associated `.prototxt` layer architecture. Demonstration shall only include deployment on images using [OpenCV-4](https://pypi.org/project/opencv-python/) as our tool. Common use case includes Asset tracking for features like position, landmarks & can be further geo-mapped.

* **Netflix Movie Recommendation** algorithm using [Singular Value Decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) and [Cosine-Similarity](https://en.wikipedia.org/wiki/Cosine_similarity) along with KFold Cross-Validation to predict all ratings given by customers to movies. Dataset is sourced from [Netflix](https://archive.org/download/nf_prize_dataset.tar) account. Primary tools utilized are [XGBoost v0.82](https://pypi.org/project/xgboost/) and [Surprise v0.1](https://pypi.org/project/surprise/). For understanding underlying statistical concepts, refer [Article-1](https://medium.com/@neuralnets/probability-distribution-statistics-for-deep-learning-73a567e65dfa) & [Article-2](https://medium.com/machine-learning-bootcamp/introduction-to-deep-learning-part-1-cbfda681c5ff) on my blog.
