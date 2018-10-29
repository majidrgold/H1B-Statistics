#!/usr/bin/python
# coding: utf-8

# In[1]:


#part 1

import csv
#import numpy as np
from collections import defaultdict
import sys
maxInt = sys.maxsize
decrement = True
while decrement:
# decrease the maxInt value by factor 10 
# as long as the OverflowError occurs.

    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True


# In[2]:


def remover(A):
    if '"' not in A:
        return A
    else:
        B = A
        C = ""
        while('"' in B):
            m = B.find('"')
            n = B[m+1:].find('"')
            C = C + B[:m+1] + B[m+1:m+n+2].replace(';','')
            B = B[m+n+2:]
            #print(C,"-",B)
        return C+B


# In[3]:


States_counter = defaultdict(int)
Job_counter = defaultdict(int)

def getData(filename1,stateIndex, jobIndex):
    CerCount = 0
    
    with open(filename1, "r",encoding="utf8") as csv1:
        reader = csv.reader(csv1,delimiter='\t') #

        counter = 0
        for row in reader:
            counter += 1
            string = ' '.join(row)
            string = remover(string)
            tempList = string.split(';')
            
            if 'CERTIFIED' not in tempList:
                continue
            else: 
                CerCount += 1
            #i = 1
            #while (i < len(tempList0)):
                #if tempList[i][0] == ' ':
                    #tempList = tempList[:i]+(' '.tempList[i:i+2])+tempList[i+2]
                #i = i + 1
            if len(tempList[stateIndex]) not in [2]:
                #print(counter,":",tempList[stateIndex])
                yield tempList,len(tempList),tempList[stateIndex] #np.array(row)
                # This will give arrays of floats, for other types change dtype
            else:
                States_counter[tempList[stateIndex]] += 1

            Job_counter[tempList[jobIndex]] += 1

    #print(CerCount)


# In[4]:


with open("../input/h1b_input.csv", "r",encoding="utf8") as csv1:
    reader = csv.reader(csv1,delimiter='\t') #
    string = ' '.join(next(reader))
    head = string.split(';')
    #print(list(enumerate(head)))
    for i in range(len(head)):
        if 'work' in head[i].lower() and 'state' in head[i].lower():
            stateIndex = i
    for i in range(len(head)):
        if 'soc' in head[i].lower() and 'name' in head[i].lower():
            jobIndex = i        
    #stateIndex = head.index('EMPLOYER_STATE')
    #jobIndex = head.index("JOB_TITLE")
    #print(stateIndex,jobIndex)

#part 2
lineCount = 0
for tup in getData("../input/h1b_input.csv",stateIndex, jobIndex):
    lineCount += 1


# In[5]:


sorted_by_value = sorted(Job_counter.items(), key=lambda x: (-x[1],x[0]), reverse = False)
total = sum([j[1] for j in sorted_by_value])
sorted_by_value

with open('../output/top_10_occupations.txt','w') as out:
    out.write('TOP_OCCUPATIONS;NUMBER_CERTIFIED_APPLICATIONS;PERCENTAGE\n')
    for i in range(min(len(sorted_by_value),10)):
        percent = round(100*sorted_by_value[i][1]/total, 1)
        out.write('{};{};{}%\n'.format(sorted_by_value[i][0].strip('"\''),sorted_by_value[i][1],percent))
        #SOFTWARE DEVELOPERS, APPLICATIONS;6;60.0%


# In[6]:


sorted_by_value_states = sorted(States_counter.items(), key=lambda x: (-x[1],x[0]), reverse = False)
total_states = sum([j[1] for j in sorted_by_value_states])

with open('../output/top_10_states.txt','w') as out:
    out.write('TOP_STATES;NUMBER_CERTIFIED_APPLICATIONS;PERCENTAGE\n')
    for i in range(min(len(sorted_by_value_states),10)):
        percent = round(100*sorted_by_value_states[i][1]/total_states, 1)
        out.write('{};{};{}%\n'.format(sorted_by_value_states[i][0].strip('"\''),sorted_by_value_states[i][1],percent))
        #SOFTWARE DEVELOPERS, APPLICATIONS;6;60.0%

