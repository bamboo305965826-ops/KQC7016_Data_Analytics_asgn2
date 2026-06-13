import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt

np.random.seed(42)
n=500
patient_id=np.arange(1,n+1)
age=np.random.randint(18,91,n)
gender=np.random.choice(['Male','Female'],n)
department=np.random.choice(['Emergency','Cardiology','General Medicine','Pediatrics','Orthopedics'],n,p=[0.24,0.22,0.24,0.15,0.15])
admission=np.random.choice(['Emergency','Walk-in','Referral','Follow-up'],n,p=[0.22,0.42,0.22,0.14])
chronic=np.random.binomial(1, np.clip(0.15+(age-18)/120,0.1,0.65))
diabetes=np.random.binomial(1, np.clip(0.08+(age-18)/150+0.15*chronic,0.05,0.65))
hypertension=np.random.binomial(1, np.clip(0.10+(age-18)/130+0.2*chronic,0.06,0.7))
# vitals
hr=np.random.normal(82,14,n) + (department=='Emergency')*12 + chronic*5
systolic=np.random.normal(122,18,n) + hypertension*22 + (department=='Cardiology')*8
oxygen=np.random.normal(97,2.3,n) - (department=='Emergency')*2.0 - chronic*1.0 - (age>70)*1.2
temp=np.random.normal(36.9,0.6,n) + (department=='Emergency')*0.7
pain=np.random.randint(1,11,n) + (department=='Emergency')*2 + (department=='Orthopedics')*1
pain=np.clip(pain,1,10)
hr=np.clip(np.round(hr),50,150).astype(int)
systolic=np.clip(np.round(systolic),90,210).astype(int)
oxygen=np.clip(np.round(oxygen,1),86,100)
temp=np.clip(np.round(temp,1),35.5,40.5)

score=(age>=65).astype(int)*1 + chronic*2 + diabetes + hypertension + (hr>=110).astype(int)*2 + (systolic>=160).astype(int)*2 + (oxygen<94).astype(int)*3 + (temp>=38).astype(int) + (pain>=8).astype(int) + (department=='Emergency').astype(int)*2 + (admission=='Emergency').astype(int)*2 + (department=='Cardiology').astype(int)
# add noise but keep clinical patterns
# no noise added for clearer decision-tree validation
risk=np.where(score>=7,'High',np.where(score>=4,'Medium','Low'))

# service generation probability helpers
lab=np.random.rand(n) < (0.35 + 0.25*(risk!='Low') + 0.15*(temp>=38) + 0.10*(department=='Emergency'))
ecg=np.random.rand(n) < (0.12 + 0.52*(department=='Cardiology') + 0.36*(risk=='High') + 0.20*(department=='Emergency'))
xray=np.random.rand(n) < (0.16 + 0.45*(department=='Orthopedics') + 0.18*(department=='Emergency') + 0.12*(pain>=8))
pharmacy=np.random.rand(n) < (0.35 + 0.28*(lab) + 0.12*(risk!='Low'))
observation=np.random.rand(n) < (0.12 + 0.52*(risk=='High') + 0.22*(admission=='Emergency') + 0.10*(department=='Emergency'))
hospital_admission=np.random.rand(n) < (0.05 + 0.45*(risk=='High') + 0.12*observation)

# ensure some rules
observation = observation | ((risk=='High') & (department=='Emergency') & (np.random.rand(n)<0.45))
ecg = ecg | ((risk=='High') & (department=='Cardiology') & (np.random.rand(n)<0.65))
lab = lab | ((risk=='High') & (np.random.rand(n)<0.55))

# satisfaction decreases with risk and high service burden
sat=np.random.normal(82,9,n)-12*(risk=='Medium')-25*(risk=='High')-5*observation-5*hospital_admission
sat=np.clip(np.round(sat,1),5,100)

df=pd.DataFrame({
 'Patient_ID': patient_id,
 'Age': age,
 'Gender': gender,
 'Department': department,
 'Admission_Type': admission,
 'Chronic_Disease': np.where(chronic==1,'Yes','No'),
 'Diabetes': np.where(diabetes==1,'Yes','No'),
 'Hypertension': np.where(hypertension==1,'Yes','No'),
 'Heart_Rate': hr,
 'Systolic_BP': systolic,
 'Oxygen_Saturation': oxygen,
 'Temperature': temp,
 'Pain_Score': pain,
 'Lab_Test': lab.astype(int),
 'ECG': ecg.astype(int),
 'X_Ray': xray.astype(int),
 'Pharmacy': pharmacy.astype(int),
 'Observation': observation.astype(int),
 'Hospital_Admission': hospital_admission.astype(int),
 'Risk_Level': risk,
 'Satisfaction_Score': sat
})
df.to_csv('/mnt/data/smart_hospital_partb_patient_risk_dataset.csv',index=False)

# EDA plots
plt.figure(figsize=(6.8,4.2))
order=['Low','Medium','High']
counts=df['Risk_Level'].value_counts().reindex(order)
counts.plot(kind='bar')
plt.title('Patient Risk Level Distribution')
plt.xlabel('Risk Level')
plt.ylabel('Number of Patients')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('/mnt/data/partb_fig_risk_distribution.png',dpi=200)
plt.close()

plt.figure(figsize=(6.8,4.2))
import numpy as np
pivot=pd.crosstab(df['Department'], df['Risk_Level']).reindex(columns=order)
pivot_pct=pivot.div(pivot.sum(axis=1),axis=0)*100
pivot_pct.plot(kind='bar', stacked=True, figsize=(7,4.2))
plt.title('Risk Level by Department')
plt.xlabel('Department')
plt.ylabel('Percentage of Patients')
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.savefig('/mnt/data/partb_fig_risk_by_department.png',dpi=200)
plt.close()

# Model
X=df[['Age','Gender','Department','Admission_Type','Chronic_Disease','Diabetes','Hypertension','Heart_Rate','Systolic_BP','Oxygen_Saturation','Temperature','Pain_Score']]
y=df['Risk_Level']
cat=['Gender','Department','Admission_Type','Chronic_Disease','Diabetes','Hypertension']
num=[c for c in X.columns if c not in cat]
# onehot handle sklearn versions
try:
    ohe=OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    ohe=OneHotEncoder(handle_unknown='ignore', sparse=False)
pre=ColumnTransformer([('cat',ohe,cat),('num','passthrough',num)])
clf=DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=42)
pipe=Pipeline([('preprocess',pre),('tree',clf)])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
pipe.fit(X_train,y_train)
pred=pipe.predict(X_test)
acc=accuracy_score(y_test,pred)
pr,rec,f1,supp=precision_recall_fscore_support(y_test,pred,labels=order,zero_division=0)
report=pd.DataFrame({'Risk_Level':order,'Precision':pr,'Recall':rec,'F1-score':f1,'Support':supp})
report.loc[len(report)] = ['Weighted Avg', *precision_recall_fscore_support(y_test,pred,average='weighted',zero_division=0)[:3], len(y_test)]
report.to_csv('/mnt/data/partb_decision_tree_metrics.csv',index=False)
print('accuracy',acc)
print(report)
print(confusion_matrix(y_test,pred,labels=order))

# Confusion matrix plot
cm=confusion_matrix(y_test,pred,labels=order)
fig, ax=plt.subplots(figsize=(5.4,4.5))
im=ax.imshow(cm)
ax.set_xticks(range(3)); ax.set_xticklabels(order)
ax.set_yticks(range(3)); ax.set_yticklabels(order)
ax.set_xlabel('Predicted Risk Level')
ax.set_ylabel('Actual Risk Level')
ax.set_title('Decision Tree Confusion Matrix')
for i in range(3):
    for j in range(3):
        ax.text(j,i,str(cm[i,j]),ha='center',va='center')
plt.tight_layout()
plt.savefig('/mnt/data/partb_fig_confusion_matrix.png',dpi=200)
plt.close()

# Tree plot
# get feature names
feature_names=list(pipe.named_steps['preprocess'].get_feature_names_out())
plt.figure(figsize=(14,7))
plot_tree(pipe.named_steps['tree'], feature_names=feature_names, class_names=pipe.named_steps['tree'].classes_, filled=True, rounded=True, fontsize=7)
plt.title('Decision Tree for Patient Risk Classification')
plt.tight_layout()
plt.savefig('/mnt/data/partb_fig_decision_tree.png',dpi=200)
plt.close()

# Association rules implementation
item_cols=['Lab_Test','ECG','X_Ray','Pharmacy','Observation','Hospital_Admission']
transactions=[]
for _,r in df.iterrows():
    items=[]
    items.append('Risk_'+r['Risk_Level'])
    items.append('Dept_'+r['Department'].replace(' ','_'))
    items.append('Admission_'+r['Admission_Type'].replace(' ','_'))
    if r['Age']>=65: items.append('Age_65plus')
    if r['Oxygen_Saturation']<94: items.append('Low_Oxygen')
    if r['Temperature']>=38: items.append('High_Temperature')
    if r['Heart_Rate']>=110: items.append('High_Heart_Rate')
    if r['Pain_Score']>=8: items.append('High_Pain')
    for c in item_cols:
        if r[c]==1: items.append('Service_'+c)
    transactions.append(set(items))

min_support=0.06
# compute supports for itemsets up to 3
from collections import Counter
item_counts=Counter()
for t in transactions:
    for k in [1,2,3]:
        for comb in combinations(sorted(t),k):
            item_counts[comb]+=1
support={k:v/n for k,v in item_counts.items() if v/n>=min_support}

rules=[]
for itemset,supp in support.items():
    if len(itemset)<2: continue
    items=set(itemset)
    # antecedents of size 1 or 2 only
    for rlen in range(1,len(itemset)):
        for ant in combinations(itemset,rlen):
            ant=set(ant); cons=items-ant
            if len(cons)!=1: continue
            ant_tuple=tuple(sorted(ant)); cons_tuple=tuple(sorted(cons))
            ant_supp=support.get(ant_tuple, item_counts.get(ant_tuple,0)/n)
            cons_supp=support.get(cons_tuple, item_counts.get(cons_tuple,0)/n)
            if ant_supp==0 or cons_supp==0: continue
            conf=supp/ant_supp
            lift=conf/cons_supp
            if conf>=0.55 and lift>=1.05:
                rules.append({'Antecedent':', '.join(sorted(ant)), 'Consequent':', '.join(sorted(cons)), 'Support':supp, 'Confidence':conf, 'Lift':lift})
rules_df=pd.DataFrame(rules).sort_values(['Lift','Confidence','Support'], ascending=False).drop_duplicates().head(12)
rules_df.to_csv('/mnt/data/partb_association_rules.csv',index=False)
print(rules_df)
# select clinically interpretable top 6 manually if exist

plt.figure(figsize=(7.2,4.2))
# service use by risk
service_by_risk=df.groupby('Risk_Level')[item_cols].mean().reindex(order)*100
service_by_risk[['Lab_Test','ECG','Observation','Hospital_Admission']].plot(kind='bar', figsize=(7.2,4.2))
plt.title('Selected Service Use by Risk Level')
plt.xlabel('Risk Level')
plt.ylabel('Patients Using Service (%)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('/mnt/data/partb_fig_service_by_risk.png',dpi=200)
plt.close()

# summary stats
print('Risk counts')
print(counts)
print('dept risk pct')
print(pivot_pct.round(1))
print('Service by risk')
print(service_by_risk.round(1))
