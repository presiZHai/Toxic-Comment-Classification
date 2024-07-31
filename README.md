## Toxic Comments Detection

1. Import Libraries
* NumPy for numerical computations
* Pandas for data manipulation and analysis
* Matplotlib and Seaborn for data visualization
* ‘re’ for regular expressions and text processing
* WordCloud for creating word cloud visualizations, 
* Scikit-learn for machine learning model building and evaluation
* TensorFlow/Keras for deep learning models.

2. Load and read, and combine the datasets

3. Exploratory Data Analysis
 * Summary Statistics with decribe method
 * Check data types and missing values
 * Check for duplicate comments and remove them if present
 * Reindex the DataFrame after dropping duplicates
 * Check the distribution of comments, 'Toxic', column and visualise the distribution with bar plot
 * Visualise the common words in toxic and non-toxic comments with Word cloud.
 
 4. Transformation of Data
 *  Data Cleaning: 
 *  Vectorization with TF-IDF transformation
 *  Addressing Class Imbalance with SMOTE

 5. Modelling
 *  Split the the resampled data into training and test sets
 *  Instantiate the deep learning model: A lightweight model, such as a simple neural network with an input layer, a hidden layer containing 64 neurons, and an output layer with a sigmoid activation function for binary classification, uses a dropout rate of 0.5 to help prevent overfitting. The model is optimized using the Adam optimizer with a learning rate of 0.001.
 *  Train the model using 10 epoch, batch size of 32 and validation split of 0.2
 *  Visualize the history to see the training and validation accuracy and loss.

 6. Evaluation

 What is a Vectorizer?
  * Text to Numerical Conversion: A vectorizer converts text data into numerical vectors that machine learning algorithms can process. In NLP, algorithms typically work with numerical data, so we need to transform textual information into a format that the algorithms can understand.
  * Feature Extraction: Vectorizers extract features from text by representing words or phrases as numerical values. These features capture important information about the text, such as word frequency, relevance, and context.
  * Dimensionality Reduction: They can help reduce the dimensionality of the text data by creating sparse matrices where each row represents a document or text sample, and each column represents a unique word or term in the corpus.

Why Use Vectorizers?
1.  Input Format for Machine Learning Models: Machine learning models, especially those used for text classification or sentiment analysis, require numerical input. Vectorizers enable us to convert raw text data into a format that these models can process and learn from.
2.  Feature Representation: Vectorizers capture semantic information from text, such as word importance and relationships between words. This allows machine learning models to learn patterns and make predictions based on the features extracted from the text.
3.  Efficient Data Representation: They provide an efficient and compact representation of text data, especially in high-dimensional spaces where the number of unique words or features can be large. This efficiency is crucial for training and deploying machine learning models effectively.
4.  Normalization and Scaling: Vectorizers often perform normalization and scaling of the numerical values, which can improve the performance and stability of machine learning algorithms, especially when dealing with diverse text data.
5.  Handling Sparse Data: Text data is often sparse, meaning that most words or features in a document are zero or absent. Vectorizers handle sparse data efficiently, storing only non-zero values and saving memory and computational resources.# Toxic-Comment-Classification
