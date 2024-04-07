# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 12:41:30 2023

@author: Amy
"""
import math
import pandas as pd
import seaborn as sb
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import ipywidgets as widgets
from ipywidgets import Box
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sklearn.cluster import KMeans

# IMPORTAREA SETULUI DE DATE DIN CSV
df = pd.read_csv("C:\\Users\Amy\Downloads\song_data.csv")
print(df)

#------------------------------
#df.var()
#df.std()
#df.median()
#df.mean()
#df.describe()
#df.shape()
#df.head()
#df.columns[]
#df.isnull().sum()
#df.info() --> un obiect 
#df.hist()
#df["song_popularity"].count()
#18835

#------------------------------
sb.histplot(df['song_popularity'], kde = True)
plt.xlabel('Popularity', fontsize = 20)
plt.ylabel('Count', fontsize = 20)
plt.show()

#------------------------------dispersii
dispersie_popularity = np.var(df["song_popularity"])
print(dispersie_popularity)
#404.3070785582185

dispersie_loudness = np.var(df["loudness"])
print(dispersie_loudness)
#16.897069408570562
#O valoare mai mică decât -1 este oblică spre stânga; acela mai mare de 1 este înclinat spre dreapta. O valoare între -1 și 1 este simetrică.

#------------------------------deviatii standard
deviatia_standard_popularity = np.std(df["song_popularity"])
print(deviatia_standard_popularity)
#20.10738865587022

deviatia_standard_loudness = np.std(df["loudness"])
print(deviatia_standard_loudness)
#4.110604506465024
#Metoda stdev() calculează abaterea standard de la un eșantion de date. Abaterea standard este o măsură a gradului de răspândire a numerelor.
#O abatere standard mare indică faptul că datele sunt răspândite, - o abatere standard mică indică faptul că datele sunt grupate strâns în jurul mediei.


#------------------------------
#REGRESIA LINIARA
#y_estimat = constanta + b1*danceability + b2*energy + b3*liveness + b4*loudness
#grafic de regresie
predictori = ["danceability", "energy", "liveness", "loudness"]
X = df[predictori]           #variabila independenta
y = df["song_popularity"]    #variabila dependenta

grafic = sb.pairplot(df, x_vars=X, y_vars="song_popularity")
plt.show()

#antrenare pe setul de date urmator: 
lm = LinearRegression()
model = lm.fit(X, y)

print("constanta= ", model.intercept_)
print("lista coef= ", model.coef_)

#constanta=  58.490843914107195
#lista coef=  [  4.03973454 -10.39096272  -3.74644148   0.67338547]

#val_vanzari =  58.490 + 4.039*danceability -10.390*energy -3.746*liveness + 0.673*loudness

x_observat = [[0.567, 0.795 , 0.114, -4.985]]
val_prezise = model.predict(x_observat)
print(val_prezise)
#[48.73663715]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=105)

lm = LinearRegression()
model = lm.fit(X_train, y_train)
print("constanta set antrenament= ", model.intercept_)
print("lista coef set antrenament= ", model.coef_)
#constanta set antrenament=  56.68824021609961
#lista coef set antrenament=  [ 4.7686609  -9.61003032 -2.6609595   0.59668917]

#predictie pe setul de date test
predictie = model.predict(X_test)
print("predictie: " , predictie)
print("y_test: " , y_test)
# predictie:  [50.20265423 49.91811318 49.50291785 ... 51.12226326 47.95408241
#  48.06172277]
# y_test:  4094     44
# 4442     55
# 6705     58
# 10080    45
# 5431     30
#          ..
# 7881     71
# 6589     40
# 13719    77
# 18535     0
# 161      77

r_patrat = r2_score(y_true = y_test, y_pred = predictie)
print(r_patrat)
#0.014293373937514953

#------------------------------
#corelatia pe intregul set de date
corelatie = df.corr()
print(corelatie)
plt.figure(figsize=(10,6))
sb.heatmap(corelatie, annot=True, cmap="coolwarm", cbar=False,linewidth = 1, annot_kws={"size": 9})
plt.title('Corelatia variabilelor')

#------------------------------
#Relatia dintre „loudness” si „liveness”
fig = plt.figure(figsize = (10,6))
sb.relplot(x="liveness", y="loudness", kind="line", data=df)

#------------------------------esantionarea a doar 4% din setul de date cu repeezenatrea grafica pe song_popularity si accousticness
sample_df = df.sample(int(0.004*len(df)))
plt.figure(figsize = [10,6])
sb.regplot(data = sample_df, y = "song_popularity", x = "acousticness", color = "g")

#------------------------------
sb.set(rc = {'figure.figsize':(20,20)})
sb.jointplot(data=df, x="loudness", y="energy", kind="kde")

#------------------------------
reqdf=df[df['danceability'] > 0.501]
#reqdf.head()
print(reqdf)
sb.relplot(x="danceability", y="speechiness", kind="line", data=reqdf)

#------------------------------
correlation_mat = df.corr()
plt.figure(figsize=(10,6))
sb.heatmap(correlation_mat, annot=True, cmap="coolwarm", linewidth = 1, annot_kws={"size": 9})

df.drop(['song_duration_ms','audio_mode','tempo','audio_valence'],axis=1, inplace=True)

sample = df.sample(500)
sb.pairplot(sample[["danceability","loudness", "acousticness", "song_popularity"]],diag_kind="kde")

#------------------------------
#schimbarea din milisecunde in secunde a timpului 
time_signature_df=pd.get_dummies(df["time_signature"])
df = pd.concat([df,time_signature_df],axis=1)
df['audio_mode'] = np.where(df['audio_mode']=='Major', 1, 0)

df['song_duration_ms'] = df['song_duration_ms'] / 1000
df.rename(columns={'song_duration_ms': 'song_duration_s'}, inplace=True) 

#------------------------------
#SE POATE ALEGE ALT SET DE DATE PT CARE S-A CALCULAT REGRESIA LINIARA:
#### Cream OLS pentru a putea face rost de coeficient pentru a calcula linia regresiei
y_popularity = df['song_popularity']
x1_dance = df['danceability']

x_dance = sm.add_constant(x1_dance)
result_dance = sm.OLS(y_popularity,x_dance).fit()
result_dance.summary()

#### yhat reprezinta linia regresiei, cream scatter-ul si aplicam linia de regresie modelului

plt.scatter(x1_dance,y_popularity)
yhat_dance = 14.5770*x1_dance + 43.7596
fig_dance = plt.plot(x1_dance, yhat_dance, lw = 4, c = "orange", label = "regression line")
plt.xlabel("Danceability", fontsize = 20)
plt.ylabel("Popularity", fontsize = 20)
plt.show()

x1_loud = df['loudness']
x_loud = sm.add_constant(x1_loud)
result_loud = sm.OLS(y_popularity,x_loud).fit()
result_loud.summary()

plt.scatter(x1_loud,y_popularity)

yhat_loud = 0.5691*x1_loud + 57.2301

fig_loud = plt.plot(x1_loud, yhat_loud, lw = 4, c = "orange", label = "regression line")
plt.xlabel("Loudness", fontsize = 20)
plt.ylabel("Popularity", fontsize = 20)
plt.show()

### Multiple liniar regression cu dummies

y_multipleLoud = df['loudness']
x1_multipleLoud = df[['tempo', 'audio_mode']]

x_multipleLoud = sm.add_constant(x1_multipleLoud)
result_multipleLoud = sm.OLS(y_multipleLoud,x_multipleLoud).fit()
result_multipleLoud.summary()

plt.scatter(df['tempo'],y_multipleLoud, c = df['audio_mode'], cmap ='RdYlGn_r')

yhat_multipleLoudNo = -9.2604 + 0.0175 * df['tempo']
yhat_multipleLoudYes = -9.745 + 0.0175 * df['tempo']

fig_multipleLoud = plt.plot(df['tempo'], yhat_multipleLoudNo, lw = 2, c = 'orange')
fig_multipleLoud = plt.plot(df['tempo'], yhat_multipleLoudYes, lw = 2, c = 'yellow')

plt.xlabel('Tempo', fontsize = 20)
plt.ylabel('Loudness', fontsize = 20)
plt.show()

#### Verificam coeficientul Tempo-ului pentru a adauga si linia de regresie normala

y_loud = df['loudness']
x1_tempo = df['tempo']

x_tempo = sm.add_constant(x1_tempo)
result_tempo = sm.OLS(y_loud,x_tempo).fit()
result_tempo.summary()

#### Pe langa cele 2 regresii anterioare o adaugam si pe cea normala

plt.scatter(df['tempo'],y_multipleLoud, c = df['audio_mode'], cmap ='RdYlGn_r')

yhat_multipleLoud = 0.0173*x1_tempo - 9.5446

fig_test2 = plt.plot(df['tempo'], yhat_multipleLoudNo, lw = 2, c = 'orange')
fig_test2 = plt.plot(df['tempo'], yhat_multipleLoudYes, lw = 2, c = 'yellow')
fig_test2 = plt.plot(df['tempo'], yhat_multipleLoud, lw = 2, c = 'blue')

plt.xlabel('Tempo', fontsize = 20)
plt.ylabel('Loudness', fontsize = 20)
plt.show()

#### Predictions with multiple regression

x1_multiple = df[['loudness','danceability','tempo','energy']]

reg_multiple = linear_model.LinearRegression()

reg_multiple = linear_model.LinearRegression()
reg_multiple.fit(x1_multiple, y_popularity)

print(reg_multiple.coef_)

print(reg_multiple.intercept_)

reg_multiple.score(x1_multiple,y_popularity)

print(x1_multiple.shape)

r2 = reg_multiple.score(x1_multiple,y_popularity)

n = x1_multiple.shape[0]
p = x1_multiple.shape[1]

adjusted_r2 = 1 - (1 -r2)*(n - 1)/(n - p - 1)
print(adjusted_r2)

#Testez predictul, -4 reprezinta valoare pe care o dau loudness-ului, 0.3 danceability-ului
#160 tempo, 0,60 energy. Valoarea care iese din print este valoarea asteptata a popularitatii


predictedPopularity = reg_multiple.predict([[-4, 0.3, 160, 0.60]])
print(predictedPopularity)

#Variabilele dependente, după cum sugerează și numele lor, depind de modul în care sunt manipulate variabilele independente, deoarece acesta este cel mai important factor într-o investigație, care este responsabil pentru modificarea rezultatului la dorința cercetătorului.
#Variabila independentă este baza tuturor cercetărilor, care este izolabilă și manipulabilă de către persoana care conduce experimentul, în timp ce variabila dependentă este un rezultat cuantificabil și măsurabil care are ca rezultat datele manipulate.
#variabile independente: song_duration_ms, acousticness, instrumentalness, key, speechiness, audio_mode, time_signature, audio_valence
#variabile dependente: song_popularity, danceability, energy, loudness, liveness, tempo

#------------------------------
#identificarea variabilele categoriale
categorial_feature = [i for i in df.columns if df[i].dtype == "object"]
print(categorial_feature)

#sau
categorical_vars = []

for column in df.columns:
    if df[column].dtype == 'object':
        categorical_vars.append(column)
print(categorical_vars)
#exista o singura variabila categoriala: song_name
#Dacă tipul de date al unei coloane este „obiect”, aceasta este considerată o variabilă categorială și este adăugată la lista de variabile categoriale.

#------------------------------
#Verificarea numărului de rânduri unice din fiecare caracteristică

target = 'song_popularity'
features = [i for i in df.columns if i not in [target]]

nu = df[features].nunique().sort_values()
nf = []; cf = []; nnf = 0; ncf = 0; #numerical & categorical features

for i in range(df[features].shape[1]):
    if nu.values[i]<=16:cf.append(nu.index[i])
    else: nf.append(nu.index[i])

print('Setul de date are {} variabile numerice & {} variabile categoriale.'.format(len(nf),len(cf)))
#Setul de date are 11 variabile numerice & 3 variabile categoriale.
#variabilele categoriale sunt key, audio_mode, time_signature

#------------------------------
#Distribuirea valorilor numerice din data frame
print("Distribuirea valorilor numerice din data frame:")
df.hist(bins = "auto", figsize = (15,15));

#------------------------------
#Să analizăm mai întâi distribuția variabilei țintă

plt.figure(figsize=[8,4])
sb.distplot(df[target], color='g',hist_kws=dict(edgecolor="black", linewidth=2), bins=30)
plt.title('Distributia variabilei tinta')
plt.show()
#Variabila țintă pare să fie distribuită în mod normal, având o medie de aproximativ 60 de unități.

#Vizualizarea trăsăturilor categoriale

print('Vizualizarea trăsăturilor categoriale:'.center(100))

n=2
plt.figure(figsize=[15,3*math.ceil(len(cf)/n)])

for i in range(len(cf)):
    if df[cf[i]].nunique()<=8:
        plt.subplot(math.ceil(len(cf)/n),n,i+1)
        sb.countplot(df[cf[i]])
    else:
        plt.subplot(2,1,2)
        sb.countplot(df[cf[i]])
        
plt.tight_layout()
plt.show()

#variabile numerice: song_popularity, song_duration_ms, danceability, acousticness, instrumentalness, energy, loudness, liveness, speechiness,  tempo, audio_valence
#variabile categoriale: key,  audio_mode, time_signatures

#------------------------------
#variabile dummy
X = df[["song_popularity", "song_duration_ms", "danceability", "acousticness", "instrumentalness", "energy", "loudness", "liveness", "speechiness", "tempo", "audio_valence" ]]
y = df["key"]

X_cu_dummy = pd.get_dummies(data = X)

X_cu_dummy_cu_drop = pd.get_dummies(data = X, drop_first = True)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train,Y_test = train_test_split(X_cu_dummy, y,test_size = 0.3, random_state=105)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_train, Y_train)
print("constanta" + model.intercept_)

df_coef = pd.DataFrame(model.coef_, X_cu_dummy.columns, columns=["Coeficienti"])

predictii = model.predict(X_test)

#comparatie liniara intre val.prezise si cele de test
sb.regplot(X=X_test, y=Y_test)

#coeficinetul de determinare
r2 = model.score(X=X_test, y=Y_test)
print(r2)
#0.000699225291926453

import statsmodels.api as sm
X_train_SM = sm.add_constant(X_train)

ls = sm.OLS(Y_train, X_train_SM).fit()
print(ls.summary())

#din punct de vedere statistic sunt semnificative toate coloanele care sunt mai mici decat 0.05, adica r patrat ajustat
#------------------------------CURS 12
X = ["danceability", "energy", "liveness", "loudness"]
y = df["song_popularity"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=100)

reg = LinearRegression()
reg.fit(X, y)
predictions = reg.predict(X)

model_LR = LinearRegression()
model_LR.fit(X_train,y_train)

y_prezis=model_LR.predict(X_test)
print("r^2=", r2_score(y_test,y_prezis))
# r^2= 0.02954530704217162

model_Ridge = Ridge(alpha=50)
model_Ridge.fit(X_train,y_train)

y_prezis_Ridge=model_Ridge.predict(X_test)
print("r^2 _ alpha=50", r2_score(y_test,y_prezis_Ridge))
print(model_Ridge.coef_)
# r^2 _ alpha=50 0.029087075089416436
# [  8.40092446 -13.32533126  -3.74359484   1.0921587 ]


model_EN=ElasticNet(alpha=10,l1_ratio=0.5)
model_EN.fit(X_train,y_train)

y_prezis_EN=model_EN.predict(X_test)
print("r^2 _ alpha =1:", r2_score(y_test,y_prezis_EN))
print(model_EN.coef_)
# r^2 _ alpha =1: 0.004758502634840456
# [ 0.         -0.         -0.          0.16796085]

model_Lasso=Lasso(alpha = 1)
model_Lasso.fit(X_train, y_train)

y_prezis_Lasso = model_Lasso.predict(X_test)
print("r^2_alpha=1:",r2_score(y_test,y_prezis_Lasso))
print(model_Lasso.coef_)
# r^2_alpha=1: 0.009378659408440293
# [ 0.         -0.         -0.          0.50210631]

#------------------------------CLUSTERE
x_cluster= df.iloc[:,4:6]
print(x_cluster)

#### WCSS folosit pentru elbow method

wcss=[]

for i in range(1,9):
   kmeans = KMeans(i)
   kmeans.fit(x_cluster)
   wcss_iter = kmeans.inertia_
   wcss.append(wcss_iter)

#### Elbow Method pentru a ne da seama de valoare kmeans

number_clusters = range(1,9)
plt.plot(number_clusters,wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(3)

kmeans.fit(x_cluster)

identified_clusters = kmeans.fit_predict(x_cluster)
print(identified_clusters)

#Adaugam tabela de cluster pentru a putea sa facem graficul

data_cluster = df.copy()
data_cluster['Cluster'] = identified_clusters
print(data_cluster)

#Facem graficul

plt.scatter(data_cluster['danceability'], data_cluster['energy'], c = data_cluster['Cluster'], cmap = 'rainbow')
plt.show()

#------------------------------COMPLETARE DUPA FEEDBACK-UL PRIMIT
df.drop_duplicates(subset=['song_name'], keep='first',inplace=True)

most_popular = df.query('song_popularity>80', inplace=False).sort_values('song_popularity', ascending=False)
print(most_popular[:10])

popular = df.query('song_popularity>40'and'song_popularity<80', inplace=False).sort_values('song_popularity', ascending=False)
print(popular[:10])


least_popular = df.query('song_popularity>0'and'song_popularity<40', inplace=False).sort_values('song_popularity', ascending=False)
print(least_popular[:10])


#------------------------------
df_1=df.groupby('song_popularity')['danceability'].mean().sort_values(ascending=[False]).reset_index()
#df_1.head()

import plotly_express as px
from plotly.offline import plot

#------------------------------  
fig2 = px.scatter(df, x="song_popularity", y="danceability", color="danceability",size='song_popularity')
plot(fig2)

#------------------------------ 
from scipy.stats import pearsonr
data1 = df_1['song_popularity']
data2 = df_1['danceability']

#corelatia lui Pearson
corr, _ = pearsonr(data1, data2)
print('corelatia lui Pearson',  corr)
#corelatia lui Pearson: 0.686

#------------------------------ 
df_2=df.groupby('song_popularity')['instrumentalness'].mean().sort_values(ascending=[False]).reset_index()

data3 = df_2['song_popularity']
data4 = df_2['instrumentalness']

#corelatia lui Pearson
corr, _ = pearsonr(data3, data4)
print('corelatia lui Pearson',  corr)
#corelatia lui Pearson -0.7459136985575765

#------------------------------  
fig3 = px.scatter(df, x="song_popularity", y="instrumentalness", color="instrumentalness",size='song_popularity')
plot(fig3)






