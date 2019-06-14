
# coding: utf-8

# 
# # Project: Investigate a Dataset (TMDb movie data)
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# The project is to do the data analysis steps to a movie dataset, and find an answers for my questions.
# 
# 
# The question that I am trying to answer with this analysis is:
# 
# 
# 1-What is the highest rating genre of all time in this data set?
# 
# 
# 2-Relationship between budget and voting average (rating)?
# 
# 
# 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# ### General Properties

# In[2]:


df=pd.read_csv('movies.csv') # load csv
df.head()


# In[3]:


#droping all unnecessary columns for my research
df.drop(['id','imdb_id','homepage','tagline','overview','production_companies','release_date','director','cast','keywords','runtime','budget_adj','revenue_adj','popularity','vote_count','revenue'],axis=1,inplace = True)


# In[4]:


df.head()


# In[5]:


#budget: the movies' budget.
#genres: is the movies' categories.
#vote average:is the average votes between the people that watched the movie.
#release year: is the release year of the movie.
#original title: name of the movie.


# In[6]:


df.shape # to find the number of columns and row.


# In[7]:


df.info() #general info about the dataframe, I want to know the types of the data.


# In[8]:


df.describe() #describtion about the dataset, we can see there is some movies with run time = 0, budget=0,revenue=0?


# In[9]:


df.hist(figsize=(8,8));


# In[10]:


sum(df.duplicated()) #to find the number of duplicated rows.


# In[11]:


df.isna().sum()  #to find the number of null values in the dataset.


# ### Data Cleaning (after finding the flows of the dataset it's time to clean it)

# In[12]:


df.dropna(inplace=True) #droping all the null values since we can't deal with null genre column.


# In[13]:


df.isna().sum() #checking the dataframe that everything is as expected.


# In[14]:


df.drop_duplicates(inplace=True) #drop all duplicates


# In[15]:


sum(df.duplicated()) #checking the dataframe that everything is as expected.


# In[16]:


#removing the data with 0 budget since it's a quality flow, also replacing it with the mean will affect the data negatively
#since it's almost half the data is missing the budget column.
zeroValues=df.query('budget==0')
zeroValues.count()


# In[17]:


df=df.drop(zeroValues.index)


# In[18]:


df.shape #checking the final shape of the data frame.


# In[19]:


df.describe() #seems more normal now!.


# In[20]:


df.dtypes #checking the types of every column. no change needed.


# In[21]:


df.hist(figsize=(8,8));


# In[22]:


# now i'm going to split the data frame each row containing 1 genre instead of all of them together
split=df[df['genres'].str.contains('|')]
split


# In[23]:


#creating a new df to put the spiltted version of the df
spiltted_df = pd.DataFrame(columns = ['release_year', 'vote_average','genres','original_title','budget'])
spiltted_df


# In[24]:


#this method will take 1 row and split it to multiple rows for each "genres" in the movies
#making it easier to deal with.
def splitting(row,length): 
    temp = 0
    global spiltted_df #our new dataframe that have the genre splitted
    while temp < length: #looping for each genre in the set of genres that are separated with pipe "|".
        new = row.copy() #making a copy of the row
        new['genres'] = row['genres'].split("|")[temp] #taking one part of the row with index i meaning; the first genre or the secound genre etc.
        spiltted_df=spiltted_df.append(new) # appending it to the newlist
        temp+=1 #increment


# In[25]:


i=0
size = df.shape[0] #size will be the whole df
while i < size: #for each row in the data frame
    splitting(split.iloc[i],len(split['genres'].iloc[i].split("|"))) # send the row and it's length to the function splitting.
    i+=1 #increment


# In[26]:


spiltted_df #taking a look at the df.


# In[27]:


spiltted_df.shape


# In[28]:


df.shape #checking the shape after splitting.


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# ### Research Question 1 (What is the highest rating genre of all time in this data set?)

# In[29]:


genre_c = spiltted_df.groupby('genres', as_index=False).count() # we grouped the data each one with it's genre and count them
genre_c

gener_s = spiltted_df.groupby('genres', as_index=False).sum() # we grouped the data each one with it's genre and took the sum
gener_s


# In[30]:


genre_c


# In[31]:


merge = genre_c.merge(gener_s,left_on='genres', right_on='genres') # merging the two data frames
merge.drop(['release_year','original_title','budget'],axis=1,inplace=True) # droping the duplicated columns
merge=merge.rename(index=str,columns={'vote_average_y':'sum_genre','vote_average_x':'count_genre'}) # renaming it for clearer names


# In[32]:


merge['sum_genre/count_genre'] = np.where(merge['count_genre'] < 1, merge['count_genre'], merge['sum_genre']/merge['count_genre']) # dividing the two columns 


# In[33]:


merge


# In[34]:


# now visualizing the data to answer the question
x=['Action','Adventure','Animation','Comedy','Crime','Documentary','Drama','Family','Fantasy','Foreign','History','Horror','Music','Mystery','Romance','Science Fiction','TV Movie','Thriller','War','Western']
plt.plot(x,merge['sum_genre/count_genre']);
fig = plt.gcf();
fig.set_size_inches(25,9)
plt.xlabel('genre')
plt.ylabel('rating average')
plt.title('each genre and it\'s rating average')
plt.show()


# In[35]:


#as we can see that Documentary is the highst rated genra by the users


# ### Research Question 2 -relationship between budget and voting average-
# 
# ##### I'm going to seperate the movies to 2 dataframe.
# ##### high budget movie will have the upper qurader 75-100% 
# ##### Low budget movies will have lower qurader 0-25%

# In[36]:


df.describe()


# In[37]:


high_budget =df.budget > 40000000 # high budget movies will have 40,000,000$ or more.
df[high_budget].info()


# In[38]:


low_budget = df.budget <= 6000000 #low budget movies will have less than 6,000,000$ budget.
df[low_budget].info()


# In[39]:


df.vote_average[high_budget].hist(label = 'high_budget',alpha=0.5);
df.vote_average[low_budget].hist(label = 'low_budget', alpha=0.5);
plt.xlabel('rating average')
plt.ylabel('number of movies')
plt.title('high vs low budget in rating')
plt.show()


# In[40]:


print(df[high_budget].mean()) #checking the mean
print(df[low_budget].mean())


# In[41]:


#both have the same mean (6)


# In[44]:


# will, we can see that low and high budget doesn't affect the rating of the movie since it's almost the same.
# and to answer the question above: no budget doesn't affect the rating


# <a id='conclusions'></a>
# ## Conclusions
# 

# In[45]:


# other than the limitation of my exploration due to lack of statistical skills we can say about our findings that:-
# finally, after applying the analysis steps to this data set we can conclude that budget doesn't affect  the rating of a movie 
#also for this data set documentaries movies have the highest ratings.

