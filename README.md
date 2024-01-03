# Introduction :
In general, we can say e-commerce is a medium powered by the internet, where customers can access an online store to browse through, and place orders for products or services via their own devices(computers, tablets, or smartphones).
Machine Learning can play a vital role in e-commerce like sales prediction, prediction of the next order of the consumer, review prediction, sentiment analysis, product recommendations e.t.c.It can also provide services through e-commerce like voice search, image search, chatbot, in-store experience(augmented reality) e.t.c.

# Business Problem :
Olist is an e-commerce site of Brazil which provides a better platform to connect merchants and their product to the main marketplace of Brazil. Olist released this dataset on Kaggle in Nov 2018.The data-set has information of 100k orders from 2016 to 2018 made at multiple marketplaces in Brazil. Its features allow viewing orders from multiple dimensions: from order status, price, payment, and freight performance to customer location, product attributes and finally reviews written by customers. 

* This business is based on the interaction between consumers, Olist store, and the seller.

* At first, an order is made by the consumer on the Olist site. This order is received by Olist store, based on the information of the order (like the product category, geolocation, mode of payment e.t.c) a notification is forwarded to the sellers.

* After that product is received from the seller and delivered to the consumer within the estimated delivery time.

* Once the customer receives the product, or if the estimated delivery date is due, the customer gets a satisfaction survey by email where he can give a note for the purchase experience and write down some comments.

  <img src="https://github.com/hasan8130/Predictive-Customer-Satisfaction-Model-for-E-commerce-Orders/blob/main/readme_pics/1.PNG" width="50%" height="50%"> 

# Data Overview :

* Source:- https://www.kaggle.com/olistbr/brazilian-ecommerce
* Uploaded In the Year : 2018
* provided by : Olist Store

The data is divided into multiple datasets for better understanding and organization.

Data is available in 9 csv files:
1. olist_customers_dataset.csv (data)
2. olist_geolocation_dataset.csv(geo_data)
3. olist_order_items_dataset.csv(order_itemdata)
4. olist_order_payments_dataset.csv(pay_data)
5. olist_order_reviews_dataset.csv(rev_data)
6. olist_orders_dataset.csv(orders)
7. olist_products_dataset.csv(order_prddata)
8. olist_sellers_dataset.csv(order_selldata)
9. product_category_name_translation.csv(order_prd_catdata)
    
 <img src="https://github.com/hasan8130/Predictive-Customer-Satisfaction-Model-for-E-commerce-Orders/blob/main/readme_pics/2.PNG" width="50%" height="50%"> 
 

 The data is merged accordingly after cleaning and analysis to get the final data needed for further preprocessing ,feature selection, and model training.
 
 I have made use of this final merged data.
 
# Defining the ML Problem Statement :

"What is the probable score that we are getting from customers?"
For a given historical data of the customer we have to predict the review score for the next order or purchase . This will help us to optimise order fulfillment and enhance customer satisfaction .

Our main hypothesis is that the product and how the order was fulfilled might influence the customer review score.Each feature we create is a new hypothesis we are testing.


# EDA :
The aim of the project is not to dive deep into EDAs (Exploratory Data Analysis) so i have made use of some publicly available notebooks for drawing conclusions. 

Some of the notebooks to refer to :
* [Data Cleaning, Viz and Stat Analysis on e-com by Azim Salikhov](https://www.kaggle.com/code/goldendime/data-cleaning-viz-and-stat-analysis-on-e-com/notebook)
* [E-Commerce Exploratory Analysis by Aguiar](https://www.kaggle.com/code/jsaguiar/e-commerce-exploratory-analysis/notebook)
*  [Olist E-commerce Data Analysis](https://medium.com/@tolamoye/olist-e-commerce-data-analysis-dff46d41e5a5)

The analysis ,helps us to understand what exactly is happening with the data .

Some conclusions that are helpful to us:
*  Product description length, photos quantities and delivery time have positive relationship with review score .
*  more expensive items sold online do have negative relationship with review score.
*  order item quantity have negative relationship with review score.
*  The most popular product category on Olist is “bed bath table,” and as we examine the popularity from the most to the least popular categories, we observe a significant decrease in their sales volume. This decrease in sales volume from the most popular to the least popular categories suggests certain implications.
* It can be concluded that sellers with review scores of 4 and 5 had the highest number of orders, indicating positive sales performance. Furthermore, sellers with higher review scores also generated higher total revenue, suggesting a correlation between positive reviews and increased sales. However, it is interesting to note that sellers with a review score of 1 had the highest average revenue, indicating that other factors may influence sales performance beyond just review scores.
* Geolocations with higher customer densities indicate areas where the company has a larger customer base and potentially higher market demand.
  

      Correlation matrix -
   <img src="https://github.com/hasan8130/Predictive-Customer-Satisfaction-Model-for-E-commerce-Orders/blob/main/readme_pics/3.PNG" width="50%" height="50%">


      Feature Importance -
   <img src="https://github.com/hasan8130/Predictive-Customer-Satisfaction-Model-for-E-commerce-Orders/blob/main/readme_pics/4.PNG" width="50%" height="50%">

# Metric :
Since we have to predict the review score for the next order , we can formulate this problem as a regression type.
For this purpose some of the metrics are:
1. RMSE score 
2. MSE score 
3. R2 score

#  Data Preprocessing and Feature Engineering :
1. Drop Columns - Some columns have information about the review given by a customer (review_coment_message, review_creation_date, etc), but we don't want to use that.The experiment assumes we don't have any information about the review, so we need to predict the score before a customer writes it. There are also some columns that are unninformative to predict the customer satisfaction.
2. Splitting the Dataset - It is important that we split our data at the very beginning of our analysis. Doing that after might introduce some unwanted bias. We can try both simple split and stratified split and see which one gives better results.
3. Separate Labels From Features - We don't wanto to apply any transformation to the labels (review_score). To avoid that we just create a separate serie with labels, and drop the target column from features dataset.
   
4. Feature Engineering - We see the original data there aren't many columns that are correlated to target.

   Correlation before adding new features 
<img src="https://github.com/hasan8130/Predictive-Customer-Satisfaction-Model-for-E-commerce-Orders/blob/main/readme_pics/5.PNG" width="50%" height="50%">
It's clear that we have to create more informative features to model this problem

* Working Days Estimated Delivery Time ->
 Gets the days between order approval and estimated delivery date. A customer might be unsatisfied if he is told that the estimated time is big.
* Working Days Actual Delivery Time ->
Gets the days between order approval and delivered customer date. A customer might be more satisfied if he gets the product faster.
* Working Days Delivery Time Delta ->
The difference between the actual and estimated date. If negative was delivered early, if positive was delivered late. A customer might be more satisfied if the order arrives sooner than expected, or unhappy if he receives after the deadline
* Is Late ->
Binary variable indicating if the order was delivered after the estimated date.
* Average Product Value ->
Cheaper products might have lower quality, leaving customers unhappy.
* Total Order Value ->
If a customer expends more, he might expect a better order fulfilment.
* Order Freight Ratio ->
If a customer pays more for freight, he might expect a better service.


 Correlation after adding new features 
 
<img src="https://github.com/hasan8130/Predictive-Customer-Satisfaction-Model-for-E-commerce-Orders/blob/main/readme_pics/6.PNG" width="50%" height="50%">

5. Check for missing values - no missing values were found .
6. Dealing with Categorical and Numerical Attributes - The way we handle categorical data is very different from the transformations needed for numerical features. We will create a transformer to select only categorical or numerical features for processing.
* For categorical features we can use techniques such as Number assigning , One Hot encoding , Mean Replacement.
For Numerical features we can standardize or normalize the data , in order to eliminate the outliers .
To handle this we can have a seperate pipeline .
7. Text data Vectorization - We sometimes have to do text preprocessing before the featurization of text data . We can use different vectorization techniques to do this .Since we have text data in the Portuguese Language, we have to be careful while choosing stopwords, and while replacing and removing any special character or any word. NLTK lirary is a good fit to solve such problems .

# Baseline Model Selection :
We can start with the very basic models such as Linear and Logisitic regression and move towards more optimal one's such as Ridge , Lasso , SVM regression , Decision Trees and Random Forests.
Finally we get the best results with the Ensemble Learning methods such as XGBoost.

We can tune our model using Grid Search CV and Cross Validation .

# Pipelines for Prediction and Deployment :

* In order to build a real-world workflow for predicting the customer satisfaction score for the next order or purchase (which will help make better decisions), it is not enough to just train the model once.
* Instead, we are building an end-to-end pipeline for continuously predicting and deploying the machine learning model, alongside a data application that utilizes the latest deployed model for the business to consume.
* This pipeline can be deployed to the cloud, scale up according to our needs, and ensure that we track the parameters and data that flow through every pipeline that runs. It includes raw data input, features, results, the machine learning model and model parameters, and prediction outputs.
* We will be using ZenML to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.

  ## ZenML and MLflow-
  
ZenML empowers your business to build and deploy machine learning pipelines in a multitude of ways:
1. By offering you a framework and template to base your own work on.
2. By integrating with tools like MLflow for deployment, tracking and more.
3. By allowing you to build and deploy your machine learning pipelines easily.

 In particular, we utilize MLflow tracking to track our metrics and parameters, and MLflow deployment to deploy our model. We also use Streamlit to showcase
 how this model will be used in a real-world setting.

## Training Pipeline
The standard training pipeline consists of several steps:

* ingest_data: This step will ingest the data and create a DataFrame.
* clean_data: This step will clean the data and remove the unwanted columns.
* train_model: This step will train the model and save the model using MLflow autologging.
* evaluation: This step will evaluate the model and save the metrics -- using MLflow autologging -- into the artifact store.
  
## Deployment Pipeline
We have another pipeline, the deployment_pipeline.py, that extends the training pipeline, and implements a continuous deployment workflow. It ingests and processes input data, trains a model and then (re)deploys the prediction server that serves the model if it meets our evaluation criteria. The criteria that we have chosen is a configurable threshold on the MSE of the training. The first four steps of the pipeline are the same as above, but we have added the following additional ones:

deployment_trigger: The step checks whether the newly trained model meets the criteria set for deployment.
model_deployer: This step deploys the model as a service using MLflow (if deployment criteria is met).

In the deployment pipeline, ZenML's MLflow tracking integration is used for logging the hyperparameter values and the trained model itself and the model evaluation metrics -- as MLflow experiment tracking artifacts -- into the local MLflow backend. 

When a new pipeline is run which produces a model that passes the accuracy threshold validation, the pipeline automatically updates the currently running MLflow deployment server to serve the new model instead of the old one.


<img src="https://github.com/hasan8130/Predictive-Customer-Satisfaction-Model-for-E-commerce-Orders/blob/main/readme_pics/7.PNG" width="50%" height="50%">

## App as a service :
To round it off, we deploy a Streamlit application that consumes the latest model service from the pipeline logic.

<img src="https://github.com/hasan8130/Predictive-Customer-Satisfaction-Model-for-E-commerce-Orders/blob/main/readme_pics/8.PNG" width="50%" height="50%">

# Running the project locally :

```bash
$ git clone https://github.com/hasan8130/Predictive-Customer-Satisfaction-Model-for-E-commerce-Orders.git
```
* create your virtual environment -
  ```bash
  python.exe -m venv venv_name
  .venv\Scrips\activate
  
* install the necessary requirements using pip .
  
 ZenML comes bundled with a React-based dashboard. This dashboard allows you to observe your stacks, stack components and pipeline DAGs in a dashboard interface. To access this, you need to launch the ZenML Server and Dashboard locally, but first you must install the optional dependencies for the ZenML server:
```bash
pip install zenml["server"]
zenml up
```
<img src="https://github.com/hasan8130/Predictive-Customer-Satisfaction-Model-for-E-commerce-Orders/blob/main/readme_pics/9.PNG" width="50%" height="50%">

If you are running the run_deployment.py script, you will also need to install some integrations using ZenML:
```bash
zenml integration install mlflow -y
```

The project can only be executed with a ZenML stack that has an MLflow experiment tracker and model deployer as a component.
```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
```
You can run two pipelines as follows:
* Training pipeline -
  
  ```bash
  python run_pipeline.py

* Deployment pipeline -
  
  ```bash
  python run_deployment.py

 You can run the streamlist app using -
  ```bash
 streamlit run app.py

