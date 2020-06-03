#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#importing the dataset

df=pd.read_csv(r"C:\Users\khato\Downloads\30292_38613_bundle_archive\avocado.csv")


# In[2]:


df.head()


# In[3]:


df.info()


# In[8]:


columns1=["#","Date","AveragePrice","Total Volume","4046","4225","4770","Total Bags","Small Bags","Large Bags","XLarge Bags","type","year","region"]
df.columns=columns1 #We have just added column (#) to the coulumns to the data frame


# In[9]:


df.columns


# In[10]:


df.info()


# In[12]:


df.describe() #it will give the static view 


# In[13]:


df.dtypes


# In[14]:


df.columns


# In[15]:


df.head()


# In[16]:


df.tail()


# Pandas dataframe. corr() is used to find the pairwise correlation of all columns in the dataframe.
# 
# Any na values are automatically excluded.
# 
# For any non-numeric data type columns in the dataframe it is ignored.

# In[17]:


df.corr()


# In[21]:


f,ax=plt.subplots(figsize=(15,15))
sns.heatmap(df.corr(),annot=True,linewidths=.5,fmt=".2f")
plt.title("Avocado Correlation Map")
plt.show()


# # MATPLOTLIB

# In[25]:


#Line Plot

#Total number of avocados with PLU 4046 sold
#Total number of avocados with PLU 4225 sold


df["4046"].plot(kind='line',color='red',linestyle=":",label="4046",grid=True,alpha=0.5,figsize=(10,10))
df['4225'].plot(kind='line',color='green',linestyle=":",label=4225,grid=True,alpha=0.5,figsize=(10,10))
plt.legend()
plt.title("Data of 4046 and 4225")
plt.show()


# In[28]:


#Scatter Plot

df.plot(kind="scatter",x="AveragePrice",y="Small Bags",color="m",grid=True,linestyle="-",figsize=(10,10))
plt.title("Average price of Small Bags")
plt.show()


# In[42]:


#Histogram

df["AveragePrice"].plot(kind="hist",color="blue",bins=10,grid=True,alpha=0.65,label="AveragePrice",figsize=(10,10))
plt.legend()
plt.xlabel("AveragePrice")
plt.title("Average Price Contribution")
plt.show()


# # PANDAS

# In[43]:


series=df["Total Bags"]
print(type(series))
df1=df[["Total Bags"]]
print(type(df))


# # Filtering -->Data Frame

# In[44]:


filtre=df["Total Bags"]>35500
df[filtre]


# In[45]:


df[df["region"]=="Atlanta"]


# # Filtering--------------->logical and

# In[46]:


df[np.logical_and(df["year"]==2015,df["Total Volume"]>10)]


# In[47]:


df[(df["type"]=="conventional") & (df["AveragePrice"]<0.6)]


# # User define funuction

# In[49]:


def listfunc():
    """return the defined list1 list"""
    list1=["alex","hagi","maradona","sneijder"]
    return list1
a,h,m,s=listfunc()
print(a,h,m,s)


# # SCOPE

# In[50]:


#1
x=2
def func():
    x=3
    return x
print(x)    #global scope 
print(func()) #Local scope
 


# In[51]:


#2
x=y=4
def func():
    x=y+1
    return x
print(x)
print(func())


# In[52]:


import builtins  #Scopes provided by python
dir(builtins)


# # NESTED FUNCTION

# In[53]:


def func():
    """return value x*add"""
    def add():
        #add local variables
        x=2
        y=8
        z=x+y
        return z
    return add()**2
print(func())


# # DEFAULT AND FLEXIBLE ARGUMENTS

# In[54]:


#default arguments

def func(x,y,z=3):    #default arguments is over written
    """return x+(y+z)"""
    return z+(x*y)
print(func(2,1))


# In[56]:


#flexible arguments
 
def func2(*args):   #We are using keyword arguments 
    for i in args:
        print(i)
func2(1,2,3,4,5)


#flexible arguments  **kwargs--->>dictionary

def func3(**kwargs):
    for key, value in kwargs.items():
        print(key+":"+value)
        
func3(alex="Brazil",hagi="Romania")


# # LAMBDA FUNCTION

# In[57]:


def f(x):
    """ """
    return lambda y: y**x

square=f(2)
cube=f(3)
print(cube(1))
print(square(2))


# # LIST COMPREHENSION

# In[58]:


limit=df.AveragePrice.mean()
print(limit)
df["rating"]=["expensive" if i>limit else "cheap" for i in df.AveragePrice]
df.loc[0:25,["rating","AveragePrice"]]


# # CLEANING DATA 

# # EXPLORATORY DATA ANALYSIS

# In[59]:


print(df["type"].value_counts(dropna=False))  #Shows the number of different avocado types


# In[60]:


print(df["year"].value_counts(dropna=False)) 


# # VISUAL EXPLORATORY DATA ANALYSIS

# In[63]:


print(df.boxplot (column="AveragePrice", by="year"))


# # TIDY DATA
# 

# In[64]:


newData=df.head()
melted=pd.melt(frame=newData,id_vars="type",value_vars=["Small Bags","Total Bags"])
melted


# # CONCATENATING DATA
# 

# In[66]:


data1=df[df["Total Bags"]>15000000]
data2=df[df["Small Bags"]>10000000]
#print(data1,data2)
concData=pd.concat([data1,data2],axis=0,ignore_index=True)
concData

#print(data1,data2,concData)


# # DATA TYPES
# 

# In[67]:


df.dtypes


# In[68]:


df["year"]=df["year"].astype("float")
df.region=df.region.astype("category")
df.dtypes


# # MISSING DATA and TESTING WITH ASSERT
# 

# In[69]:


df.info()


# In[70]:


df["XLarge Bags"].value_counts(dropna =False) #no null value


# In[71]:


assert df['XLarge Bags'].notnull().all()  #return empty


# In[72]:


#Edit columns name 
#avocado.columns = avocado.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
#avocado.columns


# # VISUAL EXPLORATORY DATA ANALYSIS
# 

# In[73]:




data1=df.loc[:,["Small Bags","AveragePrice","Total Bags"]]
data1.plot()


# In[74]:


#1  subplots
data1.plot(subplots=True)
plt.show()


# In[75]:


#2 scatter plot
data1.plot(kind="scatter",x="AveragePrice",y="Small Bags")
plt.show()


# In[76]:




#3 hist plot
data1.plot(kind="hist",y="AveragePrice",bins=50,range=(0,1))


# In[77]:


#histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "AveragePrice",bins = 50,range= (0,1),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "AveragePrice",bins = 50,range= (0,1),normed = True,ax = axes[1],cumulative=True)
plt.savefig('graph.png')
plt


# # STATISTICAL EXPLORATORY DATA ANALYSIS
# 

# In[78]:


df.describe()


# # INDEXING PANDAS TIME SERIES
# 

# In[79]:


time_list = ["1992-03-08","1992-04-12"]
print(type(time_list[1])) # As you can see date is string
# however we want it to be datetime object
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))


# In[81]:




import warnings
warnings.filterwarnings("ignore")
data2 = df.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
# lets make date as index
data2= data2.set_index("date")
data2


# In[82]:




# Now we can select according to our date index
print(data2.loc["1993-03-16"])
print(data2.loc["1992-03-10":"1993-03-16"])


# # PANDAS TIME SERIES
# 

# In[83]:


data2.resample("A").mean()


# In[84]:


data2.resample("M").mean()


# In[85]:


data2.resample("M").first().interpolate("linear")


# In[86]:


data2.resample("M").mean().interpolate("linear")


# # MANIPULATING DATA FRAMES WITH PANDAS
# 

# In[87]:


df=df.set_index("#")
df.head()


# In[88]:


df["AveragePrice"][1]


# In[89]:


df.AveragePrice[1]


# In[90]:


df.loc[1,["AveragePrice"]]


# In[91]:


df[["AveragePrice","Total Bags"]]


# # SLICING DATA FRAME
# 

# In[93]:


print(type(df["Total Bags"]))  #series
print(type(df[["Total Bags"]]))  #data frames


# In[94]:


df.loc[::,"AveragePrice":"Total Bags"]


# In[95]:


df.loc[::-1,"AveragePrice":"Total Bags"]


# In[96]:


df.loc[:,"Large Bags":]


# # FILTERING DATA FRAMES
# 

# In[97]:




#1
filter1=df["AveragePrice"]>3
df[filter1]


# In[98]:


#2
filter2=df["AveragePrice"]>3
filter3=df["year"]==2017
df[filter2 & filter3]

	


# In[99]:




#3
df.AveragePrice[df.AveragePrice>3]


# # TRANSFORMING DATA
# 

# In[100]:


def div(n):
    return n*2

df.AveragePrice.apply(div)


# In[101]:


df.AveragePrice.apply(lambda n:n*2)


# In[102]:




df["Total"]=df["Total Bags"]+df["Total Volume"]
df.head()


# # INDEX OBJECTS AND LABELED DATA
# 

# In[103]:




print(df.index.name) #output "#"
df.index.name="index_name"
df.head()


# In[104]:




data=df.copy()
data.index=range(0,18249,1)
data.head(50)


# # HIERARCHICAL INDEXING
# 

# In[105]:




data2=df.set_index(["region","type"])
data2.head(500)


# # PIVOTING DATA FRAMES
# 

# In[106]:




dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
dframe = pd.DataFrame(dic)
dframe


# In[107]:


dframe.pivot(index="gender",columns = "treatment",values="response")


# # STACKING and UNSTACKING DATAFRAME
# 

# In[108]:




df1 = dframe.set_index(["treatment","gender"])
df1


# In[109]:


df1.unstack(level=0)


# In[110]:


df1.unstack(level=1)


# In[111]:




df2 = df1.swaplevel(0,1)
df2


# # MELTING DATA FRAMES
# 

# In[112]:




#reverse of pivoting
pd.melt(dframe,id_vars="treatment",value_vars=["response","age"])


# # CATEGORICALS AND GROUPBY
# 

# In[113]:


dframe


# In[114]:


dframe.groupby("treatment").mean()


# In[115]:


dframe.groupby("treatment").age.max()


# In[116]:


dframe.groupby("response").age.max()


# In[117]:


dframe.groupby("treatment")[["age","response"]].min()


# In[ ]:




