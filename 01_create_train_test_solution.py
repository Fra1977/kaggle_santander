#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().system('ls')


# In[6]:


import pandas as pd
import numpy as np


# In[7]:


pd.options.display.max_columns=100


# In[8]:


get_ipython().run_line_magic('time', 'train = pd.read_csv("train_ver2.csv")')
display(train.head())
display(train.shape)


# In[13]:


train.describe()


# In[112]:


train.groupby("ncodpers").ind_nuevo.count().mean()


# In[113]:


get_ipython().run_line_magic('time', 'test = pd.read_csv("test_ver2.csv")')


# In[114]:


test.shape


# In[115]:


test.groupby("ncodpers").ind_nuevo.count().mean()


# In[120]:


train.groupby("ncodpers").fecha_dato.max().head()


# In[119]:


test.groupby("ncodpers").fecha_dato.max().head()


# In[14]:


uids = train.ncodpers.unique()
display(uids.shape)


# In[ ]:





# In[15]:


#uids_test_public

np.random.seed(123456) 
sample_sizes = int(0.07*uids.shape[0] )


# In[16]:


uids_test_public = set(np.random.choice(uids, size=sample_sizes,  replace=False))
len(uids_test_public)


# In[17]:


uids_train = set(uids)
display(len(uids_train))
uids_train.difference_update(uids_test_public)
display(len(uids_train))


# In[18]:


956645 - 889680


# In[19]:


66965/956645


# In[20]:


uids_test_private = set(np.random.choice(np.array(list(uids_train)), size=sample_sizes,  replace=False))
len(uids_test_private)


# In[21]:


#uids_train = set(uids)
display(len(uids_train))
uids_train.difference_update(uids_test_private)
display(len(uids_train))


# In[22]:


assert(len(uids_train.intersection(uids_test_private))==0)
assert(len(uids_train.intersection(uids_test_public))==0)
assert(len(uids_test_public.intersection(uids_test_private))==0)


# # Slit into train / test_public and test_private sets

# In[23]:


test_public  = train[train.ncodpers.isin(uids_test_public)]
test_private  = train[train.ncodpers.isin(uids_test_private)]
train = train[~train.ncodpers.isin(uids_test_private)]
train = train[~train.ncodpers.isin(uids_test_public )]


# In[24]:


display(train.shape)
display(test_public.shape)
display(test_private.shape)


# In[25]:


display(train.ncodpers.nunique())
display(test_public.ncodpers.nunique())
display(test_private.ncodpers.nunique())


# In[44]:


66965/827329


# In[27]:


assert(len(uids_test_private.intersection(uids_test_public))==0)
assert(len(uids_test_public.intersection(uids_test_private))==0)
assert(len(uids_test_public.intersection(uids_train))==0)


# # Check metrics

# In[28]:


train.groupby("ncodpers").count()


# In[29]:


display(train.groupby("ncodpers").age.count().mean(), train.groupby("ncodpers").age.count().std())
display(test_public.groupby("ncodpers").age.count().mean(), test_public.groupby("ncodpers").age.count().std())
display(test_private.groupby("ncodpers").age.count().mean(), test_private.groupby("ncodpers").age.count().std())


# # Function to tranfer one person time series into target string

# In[30]:


len(uids_train)


# In[33]:


k=7
dfi=train[train.ncodpers==list(uids_train)[k]]
dfi


# In[34]:


dfi.sort_values("fecha_dato",  ascending=True)
dfi.columns


# In[35]:


target_cols = [ 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
dfi[target_cols]


# In[36]:


dfi[target_cols].diff(axis=0)#.max(axis=0).max()


# In[37]:


def search_target(df, uid):
    dfi=train[train.ncodpers==uid].sort_values("fecha_dato",  ascending=True)
    mx=dfi[target_cols].diff(axis=0).max(axis=0).max()
    mn=dfi[target_cols].diff(axis=0).min(axis=0).min()
    return mn,mx


# In[41]:


search_target(train, list(uids_train)[k])


# In[43]:


for i in range(100):
    display(search_target(train, uids[i])[1])


# In[45]:


i=0
display(search_target(train, uids[i]))


# In[47]:


dfi=train[train.ncodpers==uids[i]].sort_values("fecha_dato",  ascending=True)
dfi[target_cols].diff(axis=0)


# In[49]:


dfid = dfi[target_cols].diff(axis=0)
dfid[dfid>0]


# In[76]:


#dfid[dfid>0].sort_values()
a=dfid[dfid >0].stack()#.index.tolist()
a.reset_index()["level_1"].values.tolist()


# In[101]:


def df2target(df,uid):
    """ must return df of format
    ncodpers,added_products
    15889,ind_tjcr_fin_ult1
    15890,ind_tjcr_fin_ult1 ind_recibo_ult1
    15892,ind_nomina_ult1
    15893,
    etc.
    """
    #dfi = df.ncodpers==uid#.isin(ncodpers)
    dfi=df[df.ncodpers==uid].sort_values("fecha_dato",  ascending=True)
    #dfi.sort("fecha_dato", inplace=True, ascending=True)
    dfid=dfi[target_cols].diff(axis=0)
    a=dfid[dfid >0].stack()#.index.tolist()
    a=a.reset_index()["level_1"].unique().tolist()
    a=",".join([str(uid),]+a)
    return a


# In[102]:


df2target(train, uids[0])


# In[103]:


for i in range(100):
    display(df2target(train, uids[i]))


# # Create Output files

# 1. Train set

# In[104]:


get_ipython().system('s')


# In[105]:


get_ipython().system('ls')


# In[106]:


get_ipython().system('pwd')


# In[107]:


train.shape


# In[108]:


train.head()


# In[109]:


train.to_csv("train_FR.csv.gz")


# 2. test_private

# In[110]:


test_private.head()


# In[116]:


test_private.groupby("ncodpers").fecha_dato.count().mean()


# In[118]:


test_private.groupby("ncodpers").fecha_dato.max().head()


# So: next steps:
#         - recreate target on last month only, 2016-05-28
#         - remove last month 2016-05-28 from train set 
#         - save train and test sets to csv
#         - create solution file of format : 
# 
#             ncodpers,added_products
#             15889,ind_tjcr_fin_ult1
#             15890,ind_tjcr_fin_ult1 ind_recibo_ult1
#             15892,ind_nomina_ult1
#             15893,
#             etc.
#             
#          + added Usage="Public"|"Private" columns
#         - add PEdro as host: ask for Kaggle UN
#         - upload code to gitlab.
#         - do a sandbox submission.
#             
#         

# In[ ]:




