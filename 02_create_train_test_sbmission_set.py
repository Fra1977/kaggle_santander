#!/usr/bin/env python
# coding: utf-8

# In[67]:


from init import *


# In[68]:


pd.options.display.max_columns


# In[177]:


get_ipython().run_line_magic('time', 'train = pd.read_csv("train_ver2.csv")')
display(train.head())
display(train.shape)


# In[182]:


train_uid = set(train.ncodpers.unique())


# In[183]:


len(train_uid)


# In[184]:


train.fecha_dato.value_counts()


# In[185]:


train_fdm = train.fecha_dato.max()
train_fdm


# # Split Train / Test Set

# In[186]:


#!Not yet!!
#test = train[train.fecha_dato==train_fdm].copy()
test=train.copy()


# In[187]:


display(test.shape)
display(test.fecha_dato.max())


# In[188]:


display(train.shape)
display(train.fecha_dato.max())
#train = train[train.fecha_dato<train_fdm] #not yet!
display(train.shape)
display(train.fecha_dato.max())


# ## check that all test UIDs are a subset of train UIDS

# In[189]:


train_uid = set(train.ncodpers.unique())
display(len(train_uid))
test_uid = set(test.ncodpers.unique())
display(len(test_uid))
display(test_uid.issubset(train_uid))


# In[190]:


len(test_uid.intersection(train_uid))


# # Create raindom split train private / public 

# In[191]:


help(input)


# In[192]:


rs = int(input());


# In[193]:


np.random.seed(rs)
test_private_uid = np.random.choice(list(new_test_uid), size=int(len(new_test_uid)/2), replace=False)


# In[194]:


test_public_uid = new_test_uid.difference(test_private_uid)


# In[195]:


display(len(test_public_uid))
display(len(test_private_uid))


# In[196]:


475976+475976


# In[88]:


test_public_uid.intersection(test_private_uid)


# # Modified find target: add last date

# In[197]:


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


import multiprocessing

#def sumall(value):
#    return sum(range(1, value + 1))

pool_obj = multiprocessing.Pool()
N = pool.map(partial(df2target, b=second_arg), a_args)
answer = pool_obj.map(sumall,range(0,5))
print(answer)


# In[152]:


get_ipython().run_cell_magic('time', '', 'test_public_data = []\nfor uid in list(test_public_uid):\n    test_public_data.append(df2target(test, uid))# , test.fecha_dato.max()))')


# In[201]:


test.sort_values(["ncodpers","fecha_dato"], ascending=True, inplace=True)
test.head(20)


# In[204]:


def df2target_subset(df):
    """ 
    works with an already subset on uid df 
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
    #dfi=df[df.ncodpers==uid].sort_values("fecha_dato",  ascending=True)
    #dfi.sort("fecha_dato", inplace=True, ascending=True)
    dfid=df[target_cols].diff(axis=0)
    dfid = dfid.iloc[-1:,:]#[dfid.fecha_dato==data_max]
    a=dfid[dfid >0].stack()#.index.tolist()
    a=a.reset_index()["level_1"].unique().tolist()
    a=",".join([str(uid),]+a)
    return a


# In[205]:


get_ipython().run_cell_magic('time', '', "test_public_data = []\nfor _, subsetDF in test.groupby('ncodpers'):\n    test_public_data.append(df2target_subset(subsetDF))")


# In[ ]:


get_ipython().run_cell_magic('time', '', "test_private_data = []\nfor _, subsetDF in test.groupby('ncodpers'):\n    test_public_data.append(df2target_subset(subsetDF))")


# In[ ]:





# In[ ]:





# In[153]:


test_private_data = []
for uid in list(test_private_uid):
    test_private_data.append(df2target(test, uid))# , test.fecha_dato.max()))


# In[156]:


display(len(test_private_data))
display(test_private_data[:20])


# In[ ]:





# In[157]:


display(len(test_public_data))
display(test_public_data[:20])


# In[164]:


with open("test_public_data_soln.csv", "w") as f:
          [f.write(i+"\n") for i in test_public_data]


# In[165]:


with open("test_private_data_soln.csv", "w") as f:
          [f.write(i+"\n") for i in test_private_data]


# In[170]:


#help(pd.read_csv)#("test_public_data_soln.csv")
pd.read_csv("test_public_data_soln.csv", infer=5000)


# In[154]:


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

# In[176]:


display(train.shape)
display(train.fecha_dato.max())
display(test.shape)
display(test.fecha_dato.max())


# In[172]:


#Now since we have created the solution, we can create the actual test set 
# i.e. last date in train set and without target column 
test = train[train.fecha_dato==train_fdm].copy()


# In[173]:


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


# ## Remove test IDs without match in train from test and solutions

# In[ ]:





# # Remove target columsn from test set 

# # Write all to disk 

# In[ ]:




