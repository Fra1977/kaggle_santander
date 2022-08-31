#!/usr/bin/env python
# coding: utf-8

# In[1]:


from init import *


# In[2]:


pd.options.display.max_columns


# In[3]:


pd.options.display.max_columns=100


# In[5]:


train_10k = pd.read_parquet("train_final_10k.parquet")


# In[6]:


train_10k


# In[20]:


non_target_cols = [i for i in train_10k.columns if not i in target_cols]
non_target_cols


# In[21]:


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
    df_ret = pd.concat([dfi[non_target_cols],dfid], axis=1 )
    return df_ret
    # FR: take all rows
    #dfid = dfid.iloc[-1:,:]#[dfid.fecha_dato==data_max]
    a=dfid[dfid >0].stack()#.index.tolist()
    #a=a.reset_index()["level_1"].unique().tolist()
    #a=",".join([str(uid),]+a)
    return dfid #a


# In[22]:


train_10k[train_10k.ncodpers==15889]


# In[23]:


df2target(train_10k,15889)


# In[ ]:





# In[18]:


train_t = []
for uid in [15889,]: # list(test_public_uid)[101:131]:
    train_t.append(df2target(train_10k, uid)) # , test.fecha_dato.max()))


# In[19]:


train_t


# In[ ]:


- apply to all uid 
- get value_counts: how many churn and buy cases in each column
- 

