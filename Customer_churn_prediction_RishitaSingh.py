#!/usr/bin/env python
# coding: utf-8

# ## EDA

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


data=pd.read_csv('C:/Users/rishi/Downloads/customer_churn_large_dataset.csv')


# In[128]:


data.head()


# In[7]:


data.shape


# In[8]:


data.info()


# In[10]:


data.describe()


# In[22]:


data.duplicated().sum()


# In[62]:


data.isnull().sum()


# In[14]:


data.skew()


# In[12]:


#Z SCORE

from scipy import stats
 
z = np.abs(stats.zscore(data['Monthly_Bill']))
print(z)


# In[6]:


numeric_columns = ['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']

for column in numeric_columns:
    sns.boxplot(x=data[column])

# Define a function to handle outliers
def handle_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)

for column in numeric_columns:
    handle_outliers(data, column)


# In[62]:


#Descriptive satatistics
# Select the variables of interest and the Thalassemia outcome
variables = ['Age','Subscription_Length_Months','Monthly_Bill','Total_Usage_GB']
outcome = 'Churn'

# Create a subset of the data with the selected variables and outcome
subset = data[variables + [outcome]].copy()

# Convert the Thalassemia outcome to categorical type
subset[outcome] = subset[outcome].astype('category')

# Descriptive statistics
statistics = subset.groupby(outcome).describe().transpose()
print("Descriptive Statistics:\n", statistics)

# Box plots to visualize the distribution of variables based on the Thalassemia outcome
for variable in variables:
    sns.boxplot(x=outcome, y=variable, data=subset)
    sns.stripplot(x=outcome, y=variable, data=subset, color=".3", size=4)
    sns.despine()
    plt.title(f"{variable} vs Churn outcome")
    plt.show()


# In[20]:


sns.distplot(data['Subscription_Length_Months'],kde=True)


# In[28]:


plt.figure(figsize=(15, 6))
sns.scatterplot(data=data, x='Monthly_Bill',y='Subscription_Length_Months',hue='Churn')


# In[32]:



# Sample data (replace with your actual dataset)
churn_data = data['Churn'].value_counts()

# Create a pie chart
plt.figure(figsize=(6, 6))
plt.pie(churn_data, labels=['Not Churned', 'Churned'], autopct='%1.1f%%', colors=['lightblue', 'lightcoral'], startangle=140,shadow=True)
plt.title('Churn Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Display the pie chart
plt.show()


# In[36]:


sns.boxplot(x='Total_Usage_GB',data=data)


# In[64]:


plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Location', hue='Churn')
plt.title('Location Distribution by Churn')
plt.xlabel('Location')
plt.ylabel('Count')
plt.legend(title='Churn', loc='upper right', labels=['Not Churned', 'Churned'])
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


# In[66]:


plt.figure(figsize=(6, 4))
sns.countplot(data=data, x='Gender', hue='Churn')
plt.title('Gender Distribution by Churn')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Churn', loc='upper right', labels=['Not Churned', 'Churned'])
plt.show()


# In[20]:


sns.histplot(x ='Monthly_Bill', data = data,kde=True)


# In[57]:


plt.figure(figsize=(9,6))
sns.kdeplot(x = "Subscription_Length_Months", y = "Total_Usage_GB", hue='Gender',data = data)


# In[34]:


import plotly.express as px
# Visualizing 'Location' Distribution
fig_location = px.histogram(data, x='Location', color='Churn', title='Location Distribution by Churn', labels={'Location': 'Location'},
                             category_orders={'Location': sorted(data['Location'].unique())})
fig_location.show()


# In[50]:


plt.figure(figsize=(9,6))
sns.heatmap(data=data.corr(),annot=True)


# In[7]:


# Encoding Categorical Variables (assuming 'Gender' and 'Location' are categorical columns)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Location'] = le.fit_transform(data['Location'])


# In[27]:


data.head()  #0-not churn 1- churn


# ### FEATURE ENGINEERING

# In[ ]:


# Removing variables that will not affect the dependent variable
data.drop(columns = ['CustomerID','Name','Location'],axis = 1,inplace = True)


# In[ ]:


df['Gender'].replace({'Male' : 1 , 'Female' : 0},inplace = True) #converting categorical value


# In[73]:


# Example code to extract the month:
data['Subscription_Month'] = data['Subscription_Length_Months'] % 12


# In[74]:


X=data[['Age','Subscription_Length_Months','Monthly_Bill','Total_Usage_GB']]
y=data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[75]:


data


# In[79]:


Q1 = data['Monthly_Bill'].quantile(0.25)
Q3 = data['Monthly_Bill'].quantile(0.75)
IQR = Q3 - Q1

outliers = (data['Monthly_Bill'] < Q1 - 1.5 * IQR) | (data['Monthly_Bill'] > Q3 + 1.5 * IQR)
df_no_outliers = data[~outliers]
df_no_outliers


# In[ ]:





# ### Model training and evaluation

# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[30]:


X=data[['Age','Subscription_Length_Months','Monthly_Bill','Total_Usage_GB']]
y=data['Churn']


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)


# ### Random forest classification

# In[25]:


from sklearn.ensemble import RandomForestClassifier
# Step 2: Model Selection and Training
# Choose a classifier (e.g., RandomForestClassifier) and train it
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 3: Model Evaluation
# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


# In[81]:


print(classification_report(y_test,y_pred))


# In[85]:


cm=(confusion_matrix(y_test,y_pred))
cm


# In[87]:


sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%')


# In[ ]:





# ### KNN

# In[121]:


from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors = 11) 
knn_model.fit(X_train,y_train)
predicted_y = knn_model.predict(X_test)
accuracy_knn = knn_model.score(X_test,y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')


# In[88]:


print(classification_report(y_test,predicted_y))
cm2=(confusion_matrix(y_test,predicted_y))
cm2


# In[89]:


sns.heatmap(cm2/np.sum(cm2), annot=True, fmt='.2%')


# In[ ]:





# ### Neural Network

# In[96]:


import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# Build a simple feedforward neural network using Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_dim=X_train.shape[1]),
     tf.keras.layers.Dropout(0.2),  # Add dropout layer
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=2,validation_data=(X_test, y_test))

# Make predictions on the test set
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


# ### Prediction for deployment

# In[27]:



# Take user input for features (Age, Subscription_Length_Months, Monthly_Bill, Total_Usage_GB)
user_input = {
    'Age': float(input('Enter Age: ')),
    'Subscription_Length_Months': float(input('Enter Subscription Length (in months): ')),
    'Monthly_Bill': float(input('Enter Monthly Bill: ')),
    'Total_Usage_GB': float(input('Enter Total Usage (in GB): '))
}

# Create a DataFrame from the user input
user_df = pd.DataFrame([user_input])

# Make a prediction
churn_prediction = clf.predict(user_df)

# Interpret the prediction
if churn_prediction[0] == 1:
    print('\nPrediction: Churn')
else:
    print('\nPrediction: Not Churn')


# In[ ]:





# In[ ]:




