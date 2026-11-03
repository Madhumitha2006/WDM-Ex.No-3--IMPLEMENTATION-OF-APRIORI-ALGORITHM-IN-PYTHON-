# WDM-Ex.No-3--IMPLEMENTATION-OF-APRIORI-ALGORITHM-IN-PYTHON-
## Name: Madhu Mitha V
## Date:
**Aim:** 
To implement Apriori Algorithm in python. <br>
**Implementation Steps:**
Follow these steps to implement Apriori algorithm in Python: 

**Import the Libraries**
The first step, as always, is to import the required libraries. Execute the following script to do so: 

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from apyori import apriori 
In the script above we import pandas, numpy, pyplot, and apriori libraries. 

**Importing the Dataset** 
Now let's import the dataset and see what we're working with. Download the dataset and place it in the "Datasets" 
folder of the "D" drive (or change the code below to match the path of the file on your computer) and execute the 
following script: 
store_data = pd.read_csv('D:\\Datasets\\store_data.csv') 
Let's call the head() function to see how the dataset looks: 
store_data.head() 
<img width="1382" height="477" alt="image" src="https://github.com/user-attachments/assets/037c3089-9f22-4748-b58d-99798292ccb8" />
A snippet of the dataset is shown in the above screenshot. If you carefully look at the data, we can see that the header 
is actually the first transaction. Each row corresponds to a transaction and each column corresponds to an item 
purchased in that specific transaction. The NaN tells us that the item represented by the column was not purchased in 
that specific transaction. 
In this dataset there is no header row. But by default, pd.read_csv function treats first row as header. To get rid of this 
problem, add header=None option to pd.read_csv function, as shown below: 
store_data = pd.read_csv('D:\\Datasets\\store_data.csv', header=None) 

Now execute the head() function: 
store_data.head() 
In this updated output you will see that the first line is now treated as a record instead of header as shown below: 
Now we will use the Apriori algorithm to find out which items are commonly sold together, so that store owner can 
take action to place the related items together or advertise them together in order to have increased profit. 
**Data Preprocessing** 
The Apriori library we are going to use requires our dataset to be in the form of a list of lists, where the whole dataset 
is a big list and each transaction in the dataset is an inner list within the outer big list. Currently we have data in the 
form of a pandas dataframe. To convert our pandas dataframe into a list of lists, execute the following script: 
records = [] 
for i in range(0, 7501): 
    records.append([str(store_data.values[i,j]) for j in range(0, 20)]) <br>
**Applying Apriori** 
The next step is to apply the Apriori algorithm on the dataset. To do so, we can use the apriori class that we imported 
from the apyori library. <br>
The apriori class requires some parameter values to work. The first parameter is the list of list that you want to extract 
rules from. The second parameter is the min_support parameter. This parameter is used to select the items with 
support values greater than the value specified by the parameter. Next, the min_confidence parameter filters those 
rules that have confidence greater than the confidence threshold specified by the parameter. Similarly, 
the min_lift parameter specifies the minimum lift value for the short listed rules. Finally, the min_length parameter 
specifies the minimum number of items that you want in your rules. 
Let's suppose that we want rules for only those items that are purchased at least 5 times a day, or 7 x 5 = 35 times in 
one week, since our dataset is for a one-week time period. The support for those items can be calculated as 35/7500 = 
0.0045. The minimum confidence for the rules is 20% or 0.2. Similarly, we specify the value for lift as 3 and 
finally min_length is 2 since we want at least two products in our rules. These values are mostly just arbitrarily 
chosen, so you can play with these values and see what difference it makes in the rules you get back out. 
Execute the following script: 
association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2) 
association_results = list(association_rules) 
In the second line here we convert the rules found by the apriori class into a list since it is easier to view the results in 
this form. 
**Viewing the Results** 
Let's first find the total number of rules mined by the apriori class. Execute the following script: 
print(len(association_rules)) 
The script above should return 48. Each item corresponds to one rule. 
Let's print the first item in the association_rules list to see the first rule. Execute the following script: 
print(association_rules[0]) 
The output should look like this: 
RelationRecord(items=frozenset({'light cream', 'chicken'}), support=0.004532728969470737, 
ordered_statistics[OrderedStatistic(items_base=frozenset({'light cream'}), items_add=frozenset({'chicken'}), 
confidence=0.29059829059829057, lift=4.84395061728395)]) 
The first item in the list is a list itself containing three items. The first item of the list shows the grocery items in the 
rule. 
For instance, from the first item, we can see that light cream and chicken are commonly bought together. This makes 
sense since people who purchase light cream are careful about what they eat, hence they are more likely to buy 
chicken i.e., white meat instead of red meat i.e., beef. Or this could mean that light cream is commonly used in recipes 
for chicken. 
The support value for the first rule is 0.0045. This number is calculated by dividing the number of transactions 
containing light cream divided by total number of transactions. The confidence level for the rule is 0.2905 which 
shows that out of all the transactions that contain light cream, 29.05% of the transactions also contain chicken. Finally, 
the lift of 4.84 tells us that chicken is 4.84 times more likely to be bought by the customers who buy light cream 
compared to the default likelihood of the sale of chicken. 
The following script displays the rule, the support, the confidence, and lift for each rule in a clearer way: 
for item in association_rules: 
    # first index of the inner list 
    # Contains base item and add item 
    pair = item[0]  
    items = [x for x in pair] 
    print("Rule: " + items[0] + " -> " + items[1]) 
    #second index of the inner list 
    print("Support: " + str(item[1])) 
    #third index of the list located at 0th 
    #of the third index of the inner list 
    print("Confidence: " + str(item[2][0][2])) 
    print("Lift: " + str(item[2][0][3])) 
    print("=====================================") 

If you execute the above script, you will see all the rules returned by the apriori class. The first four rules returned by 
the apriori class look like this: 
Rule: light cream -> chicken 
Support: 0.004532728969470737 
Confidence: 0.29059829059829057 
Lift: 4.84395061728395 
===================================== 
Rule: mushroom cream sauce -> escalope 
Support: 0.005732568990801126 
Confidence: 0.3006993006993007 
Lift: 3.790832696715049 
===================================== 
Rule: escalope -> pasta 
Support: 0.005865884548726837 
Confidence: 0.3728813559322034 
Lift: 4.700811850163794 
===================================== 
Rule: ground beef -> herb & pepper 
Support: 0.015997866951073192 
Confidence: 0.3234501347708895 
Lift: 3.2919938411349285 
===================================== 
We have already discussed the first rule. Let's now discuss the second rule. The second rule states that mushroom 
cream sauce and escalope are bought frequently. The support for mushroom cream sauce is 0.0057. The confidence 
for this rule is 0.3006 which means that out of all the transactions containing mushroom, 30.06% of the transactions 
are likely to contain escalope as well. Finally, lift of 3.79 shows that the escalope is 3.79 more likely to be bought by 
the customers that buy mushroom cream sauce, compared to its default sale. 
 
## Program: 
```

import pandas as pd
from apyori import apriori

# Load the transaction dataset
store_data = pd.read_csv("store_data.csv", header=None)

# Display the first five rows
print("Dataset:")
print(store_data.head())

# Convert the dataset into a list of transactions
records = []

for i in range(len(store_data)):
    transaction = [
        str(item).strip()
        for item in store_data.iloc[i]
        if pd.notna(item) and str(item).strip() != ""
    ]
    records.append(transaction)

print("\nTotal transactions:", len(records))

# Apply the Apriori algorithm
association_rules = apriori(
    records,
    min_support=0.0045,
    min_confidence=0.2,
    min_lift=3,
    min_length=2
)

association_results = list(association_rules)

# Display the results
print("\nTotal association rules:", len(association_results))

for item in association_results:
    pair = list(item.items)
    print("\nRule:", " -> ".join(pair))
    print("Support:", round(item.support, 4))

    for rule in item.ordered_statistics:
        if rule.items_base and rule.items_add:
            print("Association:",
                  list(rule.items_base), "->", list(rule.items_add))
            print("Confidence:", round(rule.confidence, 4))
            print("Lift:", round(rule.lift, 4))

```    
## Output:
```
C:\Users\admin\OneDrive\Documents\Apriori>
python apriori.py
Dataset:
              0   ...         19
0         shrimp  ...  olive oil
1        burgers  ...        NaN
2        chutney  ...        NaN
3         turkey  ...        NaN
4  mineral water  ...        NaN

[5 rows x 20 columns]

Total transactions: 7501

Total association rules: 24

Rule: chicken -> light cream
Support: 0.0045
Association: ['light cream'] -> ['chicken']
Confidence: 0.2906
Lift: 4.844

Rule: escalope -> mushroom cream sauce
Support: 0.0057
Association: ['mushroom cream sauce'] -> ['escalope']
Confidence: 0.3007
Lift: 3.7908

Rule: pasta -> escalope
Support: 0.0059
Association: ['pasta'] -> ['escalope']
Confidence: 0.3729
Lift: 4.7008

Rule: ground beef -> herb & pepper
Support: 0.016
Association: ['herb & pepper'] -> ['ground beef']
Confidence: 0.3235
Lift: 3.292

Rule: ground beef -> tomato sauce
Support: 0.0053
Association: ['tomato sauce'] -> ['ground beef']
Confidence: 0.3774
Lift: 3.8407

Rule: olive oil -> whole wheat pasta
Support: 0.008
Association: ['whole wheat pasta'] -> ['olive oil']
Confidence: 0.2715
Lift: 4.1224

Rule: pasta -> shrimp
Support: 0.0051
Association: ['pasta'] -> ['shrimp']
Confidence: 0.322
Lift: 4.5067

Rule: frozen vegetables -> chocolate -> shrimp
Support: 0.0053
Association: ['frozen vegetables', 'chocolate'] -> ['shrimp']
Confidence: 0.2326
Lift: 3.2545
Association: ['chocolate', 'shrimp'] -> ['frozen vegetables']
Confidence: 0.2963
Lift: 3.1084

Rule: cooking oil -> ground beef -> spaghetti
Support: 0.0048
Association: ['cooking oil', 'ground beef'] -> ['spaghetti']
Confidence: 0.5714
Lift: 3.282
Association: ['cooking oil', 'spaghetti'] -> ['ground beef']
Confidence: 0.3025
Lift: 3.079

Rule: frozen vegetables -> ground beef -> spaghetti
Support: 0.0087
Association: ['frozen vegetables', 'spaghetti'] -> ['ground beef']
Confidence: 0.311
Lift: 3.1653

Rule: frozen vegetables -> milk -> olive oil
Support: 0.0048
Association: ['frozen vegetables', 'milk'] -> ['olive oil']
Confidence: 0.2034
Lift: 3.0883
Association: ['frozen vegetables', 'olive oil'] -> ['milk']
Confidence: 0.4235
Lift: 3.2684

Rule: frozen vegetables -> mineral water -> shrimp
Support: 0.0072
Association: ['mineral water', 'shrimp'] -> ['frozen vegetables']
Confidence: 0.3051
Lift: 3.2006

Rule: frozen vegetables -> olive oil -> spaghetti
Support: 0.0057
Association: ['frozen vegetables', 'spaghetti'] -> ['olive oil']
Confidence: 0.2057
Lift: 3.124

Rule: frozen vegetables -> shrimp -> spaghetti
Support: 0.006
Association: ['frozen vegetables', 'spaghetti'] -> ['shrimp']
Confidence: 0.2153
Lift: 3.0131

Rule: frozen vegetables -> tomatoes -> spaghetti
Support: 0.0067
Association: ['frozen vegetables', 'spaghetti'] -> ['tomatoes']
Confidence: 0.2392
Lift: 3.498
Association: ['tomatoes', 'spaghetti'] -> ['frozen vegetables']
Confidence: 0.3185
Lift: 3.3411

Rule: grated cheese -> ground beef -> spaghetti
Support: 0.0053
Association: ['grated cheese', 'spaghetti'] -> ['ground beef']
Confidence: 0.3226
Lift: 3.2831

Rule: mineral water -> ground beef -> herb & pepper
Support: 0.0067
Association: ['mineral water', 'herb & pepper'] -> ['ground beef']
Confidence: 0.3906
Lift: 3.9757

Rule: ground beef -> herb & pepper -> spaghetti
Support: 0.0064
Association: ['herb & pepper', 'spaghetti'] -> ['ground beef']
Confidence: 0.3934
Lift: 4.0044

Rule: milk -> olive oil -> ground beef
Support: 0.0049
Association: ['milk', 'ground beef'] -> ['olive oil']
Confidence: 0.2242
Lift: 3.4049

Rule: shrimp -> ground beef -> spaghetti
Support: 0.006
Association: ['shrimp', 'ground beef'] -> ['spaghetti']
Confidence: 0.5233
Lift: 3.0053

Rule: milk -> olive oil -> spaghetti
Support: 0.0072
Association: ['milk', 'spaghetti'] -> ['olive oil']
Confidence: 0.203
Lift: 3.0825

Rule: olive oil -> mineral water -> soup
Support: 0.0052
Association: ['mineral water', 'soup'] -> ['olive oil']
Confidence: 0.2254
Lift: 3.423

Rule: olive oil -> pancakes -> spaghetti
Support: 0.0051
Association: ['pancakes', 'spaghetti'] -> ['olive oil']
Confidence: 0.2011
Lift: 3.0529

Rule: frozen vegetables -> milk -> mineral water -> spaghetti
Support: 0.0045
Association: ['milk', 'mineral water', 'spaghetti'] -> ['frozen vegetables']
Confidence: 0.2881
Lift: 3.0228
```
 
## Result: 
Thus, the python program for implementing Apriori Algorithm has been executed successfully. f
