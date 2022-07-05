import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from statistics import geometric_mean
warnings.filterwarnings('ignore')



############################
"""
'b'       boolean
'i'       (signed) integer
'u'       unsigned integer
'f'       floating-point
'c'       complex-floating point
'O'       (Python) objects
'S', 'a'  (byte-)string
'U'       Unicode
'V'       raw data (void)
"""


## Separar tipos de columnas
## no object, variables object
def split_columns(df):
    dic_types = dict(zip(df.columns,df.dtypes))
    not_object_cols = list(dict(filter(lambda x: x[1] != 'O', dic_types.items())).keys()) 
    object_cols = [x for x in df.columns if x not in not_object_cols]

    return not_object_cols,object_cols

###########################################


## media geometrica
########################################### 
def g_mean(column):
    
    column = [x for x in column if (x >= 0)]
    a = np.log(column)
    return np.exp(a.mean())    
##########################################

## univariado numerico
########################################### 
def univariate_numerc_cols_plots(df):
    
    #tema grafico
    sns.set(style="ticks")
    
    #colores de lineas en grafico
    color_mean = 'darkred'
    color_percentil = 'darkorange'
    color_min_max = 'navy'
    
    for column in df:
    
        #figura con 2 Matplotlib.axes (ax_box, ax_hist)
        f, (ax_box,ax_hist) = plt.subplots(2,
                                   sharex= True,
                                   gridspec_kw={"height_ratios": (.30, .70)})

        ##un grafico para cada ax
        #boxplot
        sns.boxplot(df[column], 
                ax = ax_box, 
                color="white")
        #histograma
        h= sns.histplot(df,
                 x = column,
                 kde=True,
                 ax = ax_hist, 
                 color="silver")

        #resaltamos distribución del histograma
        h.lines[0].set_color('black')
        
        #cuartiles
        q1,q3 = np.percentile(df[column], [25,75])
        
        #media aritmetica
        my_mean = np.mean(df[column])
        
        #agregamos media
        ax_hist.axvline(my_mean, color=color_mean)

        # agregamos cuartiles q1,q3
        ax_hist.axvline(q1, color=color_percentil)
        ax_hist.axvline(q3, color=color_percentil)
        
        #agregamos min,max
        ax_hist.axvline(df[column].min(), color=color_min_max)
        ax_hist.axvline(df[column].max(), color=color_min_max)
        
        #calculos de información
        longitud = len(df[column])
        menor_q1 = round(((len(df.loc[df[column] < q1]))/longitud)*100,2)
        mayor_q3 = round(((len(df.loc[df[column] > q3]))/longitud)*100,2)
        
        
        plt.show()
        
        #info
        print(
f""".-. nombre:{column.upper()} .-.
INFO:
cont: {longitud}
min: {df[column].min()}   media_ar: {round(my_mean,2)}   media_geo: {round(g_mean(df[column]),2)}   moda: {df[column].mode()[0]}   desviacion:{round(df[column].std(),2)}   max:{df[column].max()}
Q1: {q1}   %debajo Q1(25): {menor_q1}% 
Q3: {q3}   %arriba Q3(75): {mayor_q3}% 
""")
        
    return None

#########################################################



####descripcion univariada categoricas
#######################################################
def univariate_categorical_cols_plots(df,max_categories):
    
    sns.set(style="ticks",rc = {'figure.figsize':(10,5)})
    
    colors= ['blue','orange','green','red','gold']
    
    total = float(len(df))
    
    print(f'****** máximas categorias: {max_categories} ******\n')
    
    for column in df:
            
        number_categorical = len(df[column].value_counts())
        
        if number_categorical > max_categories:
            print(
f"""Columna: {column.upper()}
** Demasiadas categorias para mostrar: {number_categorical} ({round((number_categorical/total)*100,2)}% valores unicos) **
"""
            )
             
        else:
            print(
f"""Columna: {column.upper()}
Numero de categorias: {number_categorical}
"""                
            )
                        
            ax = sns.countplot(x=column, data=df,palette=sns.blend_palette(colors,number_categorical))
            
            for p in ax.patches:
                percentage = '{:.1f}%'.format(100 * p.get_height()/total)
                x = p.get_x() + p.get_width()
                y = p.get_height()
                ax.annotate(percentage, (x-.3, y),ha='center')
            
            ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
            plt.show()    
        
    return None

################################################################


#heat map and dataframe with corr
################################################################
def multivariate_numerical_corr(df):
    corr_matrix = df.corr()
    
    sns.heatmap(corr_matrix,
                cmap="RdBu",
                center=0,
                vmin= -1,
                vmax= 1)
    
    upper_corr_mat = corr_matrix.where( 
    np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)) 
  
    unique_corr_pairs = upper_corr_mat.unstack().dropna() 
  
    sorted_mat = unique_corr_pairs.sort_values() 

    df_correlations = pd.DataFrame(list(zip(sorted_mat.index,list(sorted_mat))),columns=['relation','correlation_value'])

    
    return df_correlations

###################################################


