#!/usr/bin/env python
# coding: utf-8

# In[67]:


from init import *


# In[68]:


pd.options.display.max_columns


# In[69]:


get_ipython().run_line_magic('time', 'train = pd.read_csv("train_ver2.csv")')
display(train.head())
display(train.shape)


# In[70]:


train_uid = set(train.ncodpers.unique())


# In[71]:


len(train_uid)


# In[72]:


train.fecha_dato.value_counts()


# In[73]:


train_fdm = train.fecha_dato.max()
train_fdm


# # Modified find target: add last date

# In[138]:


def df2target(df,uid):
    """ 
    converts a competition data frame filteredd for a given uid into a target string 
    with the submissionformat:
    
    must return df of format
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
    dfid = dfid.iloc[-1:,:]#[dfid.fecha_dato==data_max]
    a=dfid[dfid >0].stack()#.index.tolist()
    a=a.reset_index()["level_1"].unique().tolist()
    a=",".join([str(uid),]+a)
    return a


# In[150]:


#test_public_data = []
#for uid in list(test_public_uid)[101:111]:
#test_public_data.append(df2target(test, uid))# , test.fecha_dato.max()))
#%lprun df2target(test, uid)
def testrun():
    test_public_data = []
    for uid in list(test_public_uid)[101:131]:
        test_public_data.append(df2target(test, uid))# , test.fecha_dato.max()))
get_ipython().run_line_magic('prun', '-s cumulative  -l 10 testrun()')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'test_public_data = []\nfor uid in list(test_public_uid):\n    test_public_data.append(df2target(test, uid))# , test.fecha_dato.max()))')


# In[142]:


type(a)


# In[130]:


test_public_data


# In[126]:


#5% taxa
#for i in range(100):
#    uid = list(test_public_uid)[i]
#    #uid
#    dfi=test[test.ncodpers==uid].sort_values("fecha_dato",  ascending=True)
#    dfid=dfi[target_cols].diff(axis=0)
#    dfid = dfid.iloc[-1:,:]#.reset_index()
#    a=dfid[dfid >0].stack()
#    display(a)


# # Split Train / Test Set

# In[75]:


#!Not yet!!
#test = train[train.fecha_dato==train_fdm].copy()
test=train.copy()


# In[76]:


display(test.shape)
display(test.fecha_dato.max())


# In[77]:


display(train.shape)
display(train.fecha_dato.max())
train = train[train.fecha_dato<train_fdm]
display(train.shape)
display(train.fecha_dato.max())


# ## check that all test UIDs are a subset of train UIDS

# In[78]:


train_uid = set(train.ncodpers.unique())
display(len(train_uid))
test_uid = set(test.ncodpers.unique())
display(len(test_uid))
display(test_uid.issubset(train_uid))


# In[79]:


len(test_uid.intersection(train_uid))


# In[80]:


new_test_uid = test_uid.intersection(train_uid)
test = test[test.ncodpers.isin(new_test_uid)]
display(test.shape)


# In[81]:


#test again 
train_uid = set(train.ncodpers.unique())
display(len(train_uid))
test_uid = set(test.ncodpers.unique())
display(len(test_uid))
display(test_uid.issubset(train_uid))


# # Create raindom split train private / public 

# In[82]:


help(input)


# In[83]:


rs = int(input());


# In[84]:


np.random.seed(rs)
test_private_uid = np.random.choice(list(new_test_uid), size=int(len(new_test_uid)/2), replace=False)


# In[85]:


test_public_uid = new_test_uid.difference(test_private_uid)


# In[86]:


display(len(test_public_uid))
display(len(test_private_uid))


# In[87]:


463380+463380


# In[88]:


test_public_uid.intersection(test_private_uid)


# # Create solution  from test set 

# In[64]:


test_public_data = []
for uid in list(test_public_uid):
    test_public_data.append(df2target(train, uid))


# In[66]:


test_public_data[:100]


# # Remove target columsn from test set 

# # Write all to disk 

# In[ ]:




