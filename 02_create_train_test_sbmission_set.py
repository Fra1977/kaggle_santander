#!/usr/bin/env python
# coding: utf-8

# In[1]:


from init import *


# In[2]:


pd.options.display.max_columns


# In[3]:


get_ipython().run_line_magic('time', 'train = pd.read_csv("train_ver2.csv")')
display(train.head())
display(train.shape)


# In[8]:


train_uid = set(train.ncodpers.unique())


# In[9]:


len(train_uid)


# In[10]:


train.fecha_dato.value_counts()


# In[11]:


train_fdm = train.fecha_dato.max()
train_fdm


# # Split Train / Test Set

# In[12]:


#!Not yet!!
#test = train[train.fecha_dato==train_fdm].copy()
test=train #.copy()


# In[13]:


display(test.shape)
display(test.fecha_dato.max())


# In[ ]:


display(train.shape)
display(train.fecha_dato.max())
#train = train[train.fecha_dato<train_fdm] #not yet!
display(train.shape)
display(train.fecha_dato.max())


# # Modified find target: add last date

# In[17]:


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


# In[39]:


test.sort_values(["ncodpers","fecha_dato"], ascending=True, inplace=True)
test.head(20)


# In[40]:


import pdb

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
    #pdb.set_trace()
    dfid=df[target_cols].diff(axis=0)
    dfid = dfid.iloc[-1:,:]#[dfid.fecha_dato==data_max]
    a=dfid[dfid >0].stack()#.index.tolist()
    a=a.reset_index()["level_1"].unique().tolist()
    #a=",".join([str(uid),]+a)
    a = " ".join(a)
    return a


# In[37]:


#%%time 
#test_public_data = []
#for _, subsetDF in test.groupby('ncodpers'):
#    test_public_data.append(df2target_subset(subsetDF))


# better than 2x 1.5h anyways. 

# In[42]:



for _, subsetDF in test.loc[:100].groupby('ncodpers'):
    print(str(_) +","+ df2target_subset(subsetDF) )
    #print(df2target_subset(subsetDF))
    


# In[67]:


display(test.shape)
display(test.ncodpers.nunique())


# In[68]:


test_uids = np.random.choice(test.ncodpers.unique(), size=10000, replace=False)


# In[69]:


len(np.unique(test_uids))


# In[70]:


np.sort(test_uids)[:10]


# In[71]:


test_public_data = None
test_private_data = None
import gc
gc.collect()


# In[72]:


get_ipython().run_cell_magic('time', '', '\ntest2 = test[test.ncodpers.isin(test_uids)].copy()\ntest2.sort_values(["ncodpers","fecha_dato"], ascending=True, inplace=True)\ntest_data = []\nfor _, subsetDF in test2.groupby(\'ncodpers\'):\n    test_data.append([_,df2target_subset(subsetDF) ])\n    ')


# In[73]:


test2.ncodpers.nunique()


# In[74]:


len(test_data)


# In[75]:


test_data[:100]


# In[77]:


df_subm = pd.DataFrame(test_data, columns=["ID", "Expected"])


# Now what needs to be done: 
# 
#     - get all train uids that have at least the last month 
#     - these are the test uids 
#     - split into private / public test set 
#     - go through list again, keep only test uids, convert the comma sep list into space separated,
#     add Public/Pravate Label, add columns ID, Expected, Usage , save as csv 
#     - then, split train into first months, test = last month 
#     - save both 
#     - upload to sandbox 
#     
#     
#  Then: 
#    - debug sandbox. Try MAP@1 only, remove index, add Usage column 
#    - create larger >=100k solution and test file. 

# In[78]:


df_subm.head()


# In[80]:


df_subm.shape


# In[85]:


df_subm[df_subm.Expected!=""].shape


# In[86]:


311/10000


# target rate = 311/10000 =  3%
# 

# In[79]:


train_uids_atmax = train[train.fecha_dato==train.fecha_dato.max()].ncodpers.unique()


# In[88]:


df_subm = df_subm[df_subm.ID.isin(train_uids_atmax)]


# In[89]:


df_subm


# In[90]:


df_subm.to_csv("submission.csv")


# In[93]:


sample = df_subm.sample(100)
sample.columns=["Id", "Predicted"]
sample.to_csv("submission_sample_FR.csv")


# In[ ]:





# In[ ]:


Team ToDO: 
    
    - Day 19 
    - Day 20 upload test / tran and submission set 
    - Day 21 do and test   sandbox submission
    - Day 22 create rules   / insights presentation
    - Day 23
    - Day 24 
    - Day 25
    - Day 26: competition


# In[209]:


train.ncodpers.nunique()


# In[59]:


train[train.fecha_dato==train.fecha_dato.max()].ncodpers.nunique()


# In[58]:


train_uids_atmax = train[train.fecha_dato==train.fecha_dato.max()].ncodpers.unique()


# In[94]:


#for row in test_public_data[:10]:
#    print(row)
#    rowsplit =  row.split(",")
#    uid = rowsplit[0]
#    rest = " ".join(rowsplit[1:])
#    if uid in train_uids_atmax: 
#        print("match: ", uid, rest)
#    else:
#        print("nomatch: ", uid, rest)


# **run until here to create submission and sample files.**

# In[ ]:





# In[ ]:





# ## check that all test UIDs are a subset of train UIDS

# In[99]:


train_uid = set(train.ncodpers.unique())
display(len(train_uid))
test_uid = set(df_subm.ID.unique())
display(len(test_uid))
display(test_uid.issubset(train_uid))


# In[100]:


len(test_uid.intersection(train_uid))


# In[ ]:


test_final = train[train.fecha_dato==train_fdm].copy()
#train = train[train.fecha_dato<train_fdm] #not yet!
test_final = test_final[~target_cols]


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




