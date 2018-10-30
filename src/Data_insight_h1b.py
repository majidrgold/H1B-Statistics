 
# coding: utf-8

# In[1]:


#part 1

import csv  # to handle csv files
from collections import defaultdict # to categorize states and occupations
import sys # to print results in subdirectory folder
'''The csv file might contain very huge fields; therefore, the field_size_limit should be increased:
'''
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
'''This funciton is defined to overcome the unindented semicolon's problem.
Extra semicolons exists for example in address block while semicolon is used as the sepator in the csv file. 
'''
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

'''The function gets filename1, StateIndex and jobIndex and reads csv file line by line .
It produces dictionaries of Job_counter ( key : job title, value : number of certified h1b in that job) and 
States_counter ( key : States, value : number of certified h1b in the state). 
The following function uses the remover function defined in the previous block.
The function gets filename1, StateIndex and jobIndex and computes Job_counter and State_counter.
'''
States_counter = defaultdict(int)
Job_counter = defaultdict(int)

def getData(filename1,stateIndex, jobIndex):
    CerCount = 0
    
    with open(filename1, "r") as csv1:
        reader = csv.reader(csv1,delimiter='\t') #,encoding="utf8"

        counter = 0
        for row in reader: 
            counter += 1
            string = ' '.join(row)# join one row to eliminate the unintended separation of elements
            string = remover(string)
            tempList = string.split(';')
            
            if 'CERTIFIED' not in tempList:
                continue
            else: 
                CerCount += 1
            if len(tempList[stateIndex]) not in [2]:
                #print(counter,":",tempList[stateIndex])
                yield tempList,len(tempList),tempList[stateIndex] 
                # This will give arrays of floats, for other types change type
            else:
                States_counter[tempList[stateIndex]] += 1

            Job_counter[tempList[jobIndex]] += 1

    #print(CerCount)


# In[4]:

'''This part finds the columns of work states and job names which
will be found by two key words. Work and state are keywords for work states 
and soc and name are key words for job titiles. 
'''
with open("input/h1b_input.csv", "r") as csv1:        
    reader = csv.reader(csv1,delimiter='\t') #,encoding="utf8"
    string = ' '.join(next(reader)) 
    head = string.split(';') #reads header
    #print(list(enumerate(head))) 
    for i in range(len(head)):
        if 'work' in head[i].lower() and 'state' in head[i].lower(): 
            stateIndex = i
    for i in range(len(head)):
        if 'soc' in head[i].lower() and 'name' in head[i].lower():
            jobIndex = i        

#part 2
#lineCount = 0
for tup in getData("input/h1b_input.csv",stateIndex, jobIndex):
    #lineCount += 1
    pass

# In[5]:
#This part sorts the values of certified h1b for a specific job title and measures the percentage of each job title in respect to total certified h1bs.

sorted_by_value = sorted(Job_counter.items(), key=lambda x: (-x[1],x[0]), reverse = False)
total = sum([j[1] for j in sorted_by_value])
sorted_by_value

#prints top_10_occupations.txt in output directory
with open('output/top_10_occupations.txt','w') as out:
    out.write('TOP_OCCUPATIONS;NUMBER_CERTIFIED_APPLICATIONS;PERCENTAGE\n')
    for i in range(min(len(sorted_by_value),10)):
        percent = round(100*sorted_by_value[i][1]/total, 1)
        out.write('{};{};{}%\n'.format(sorted_by_value[i][0].strip('"\''),sorted_by_value[i][1],percent))
        #SOFTWARE DEVELOPERS, APPLICATIONS;6;60.0%

# In[6]:

# this part sorts the values of certified h1bs in different states and measures the percentage of job applicant.
sorted_by_value_states = sorted(States_counter.items(), key=lambda x: (-x[1],x[0]), reverse = False)
total_states = sum([j[1] for j in sorted_by_value_states])
# prints top_10_states.txt in output directory
with open('output/top_10_states.txt','w') as out:
    out.write('TOP_STATES;NUMBER_CERTIFIED_APPLICATIONS;PERCENTAGE\n')
    for i in range(min(len(sorted_by_value_states),10)):
        percent = round(100*sorted_by_value_states[i][1]/total_states, 1)
        out.write('{};{};{}%\n'.format(sorted_by_value_states[i][0].strip('"\''),sorted_by_value_states[i][1],percent))
        #SOFTWARE DEVELOPERS, APPLICATIONS;6;60.0%

