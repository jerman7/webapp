from timeit import default_timer as timer

start = timer()

##IMPORT PACKAGES
import streamlit as st
import pandas as pd
import re
from sklearn.cluster import KMeans
import numpy as np
import pickle
##Importing language packages
import eng_to_ipa as ipa
import epitran
from pyphonetics import RefinedSoundex

##Streamlit App

st.title('PhonNameMatcher')

st.write('A tool that helps you find phonetically similar names across languages.')

firstname = st.text_input("What is a name that you like?","Type name here..")
language1 = st.text_input("What language is it in?","English, Spanish, Turkish or German..")
language2 = st.text_input("In which language would you like to see phonetically similar names?","English, Spanish, Turkish or German..")
gender = st.text_input("What is your gender preference for a name?","Male or female..")


	

##Importing data
ipa_cv = pd.read_csv('/Users/Jeylan/Documents/clbng_web/phenome_consonants_vowels_new_edited.csv')
df3 = pd.read_csv('/Users/Jeylan/Documents/clbng/Data/df3.csv')
ipa_cv_doubles = pd.read_csv('/Users/Jeylan/Documents/clbng_web/phenome_consonants_vowels_new_edited.csv')

epi_turkish = epitran.Epitran('tur-Latn') 
epi_spanish = epitran.Epitran('spa-Latn')
epi_german = epitran.Epitran('deu-Latn')


all_phenomes = ipa_cv['phoneme'].astype(str).values.tolist()
turkish_cv = ipa_cv_doubles[ipa_cv_doubles['language'] == 'turkish']['phoneme'].to_list()
spanish_cv = ipa_cv_doubles[ipa_cv_doubles['language'] == 'spanish']['phoneme'].to_list()
german_cv = ipa_cv_doubles[ipa_cv_doubles['language'] == 'german']['phoneme'].to_list()
english_cv = ipa_cv_doubles[ipa_cv_doubles['language'] == 'english']['phoneme'].to_list()
df3.set_index('Name', inplace=True)


#st.write("first part")

#end = timer()
#st.write(end - start) 


#start = timer()


@st.cache
def name_to_phenome_list(name1, all_phenomes):
    all_phenomes = sorted(all_phenomes,key=len)[::-1] # sort longest to smallest
    lookup_dict = {}
    for p in all_phenomes:
        if p in name1:
            #
            # find the start indexes where this occurs 
            # just to get the ordering right
            #
            starts = [x.start() for x in re.finditer(p,name1)]
 
            # remoev thise from the name so we don't repeat things we don't need
            # replace with blank spaces to keep length counts consistent
            # WARNING: if '_' is a character in a phenome (idk if it is) this won't work
            #          need to pick a new character
            name1 = name1.replace(p,'_'*len(p))
            
            for s in starts:
                lookup_dict[s] = p # assign to dict
    # now that we're done, recombine
    outlist = [lookup_dict[k] for k in np.sort(list(lookup_dict.keys()))]
    return outlist
    

#st.write("second part")

#end = timer()
#st.write(end - start) 



#start = timer()


df4 = df3.reset_index()
df4 = df4.drop(df4.columns[0], axis=1)
df4 = df4.iloc[0:1]
df4.iloc[0, :] = df4.replace(0, np.nan).bfill().iloc[0, :]
df4.iloc[0, :] = df4.replace(1, np.nan).bfill().iloc[0, :]
df4 = df4.fillna(0)
df4 = df4.drop(columns=['new_column', 'new_new_column', 'Gender', 'Language'])





def function(firstname, language1, language2, gender):

    lang_dict = {
            'turkish': turkish_cv,
            'Turkish': turkish_cv,
            'english': english_cv,
            'English': english_cv,
            'german': german_cv,
            'German': german_cv,
            'Spanish': spanish_cv,
            'spanish': spanish_cv}
    if language1 == "English" or language1 == 'english':
        import eng_to_ipa as ipa
        dataf= pd.DataFrame([firstname], columns = ["Name"])
        dataf['new_column'] = dataf['Name'].apply(ipa.convert, keep_punct=False)
        dataf['new_column'] = dataf['new_column'].str.replace('ˈ','')
        dataf['new_column'] = dataf['new_column'].str.replace('ˌ','')
        dataf['new_new_column'] = dataf['new_column'].apply(name_to_phenome_list, args=(all_phenomes,))  
        list_of_names = dataf['new_new_column'].to_list()    
        list_of_names = list(list_of_names[0])
    elif language1 == "Turkish" or language1 == 'turkish' :
        dataf= pd.DataFrame([firstname], columns = ["Name"])
        dataf['new_column'] = dataf['Name'].apply(epi_turkish.transliterate)
        dataf['new_column'] = dataf['new_column'].str.replace('ˈ','')
        dataf['new_column'] = dataf['new_column'].str.replace('ˌ','')
        dataf['new_new_column'] = dataf['new_column'].apply(name_to_phenome_list, args=(all_phenomes,))  
        list_of_names = dataf['new_new_column'].to_list()    
        list_of_names = list(list_of_names[0])        
    elif language1 == "Spanish" or language1 == 'spanish':
        dataf= pd.DataFrame([firstname], columns = ["Name"])
        dataf['new_column'] = dataf['Name'].apply(epi_spanish.transliterate)
        dataf['new_column'] = dataf['new_column'].str.replace('ˈ','')
        dataf['new_column'] = dataf['new_column'].str.replace('ˌ','')
        dataf['new_new_column'] = dataf['new_column'].apply(name_to_phenome_list, args=(all_phenomes,))  
        list_of_names = dataf['new_new_column'].to_list()    
        list_of_names = list(list_of_names[0])            
    elif language1 == "German" or  language1 == 'german':
        import eng_to_ipa as ipa
        dataf= pd.DataFrame([firstname], columns = ["Name"])
        dataf['new_column'] = dataf['Name'].apply(epi_german.transliterate)
        dataf['new_column'] = dataf['new_column'].str.replace('ˈ','')
        dataf['new_column'] = dataf['new_column'].str.replace('ˌ','')
        dataf['new_new_column'] = dataf['new_column'].apply(name_to_phenome_list, args=(all_phenomes,))  
        list_of_names = dataf['new_new_column'].to_list()    
        list_of_names = list(list_of_names[0])    
    if all(elem in list(set(lang_dict[language1]) & set(lang_dict[language2]))  for elem in list_of_names):
        st.write(str(firstname) + " is pronounceable in both languages.")
    else: 
        dataframe = pd.concat([dataf, df4.reindex(dataf.index)], axis=1)
        for i in range(len(dataframe)): # loop over all rows
                  # assuming this column contains an array / list of the phenomes
                    # that is iterable:
            phenomes = dataframe.iloc[i]['new_new_column']
            for p in phenomes: 
                dataframe.at[i,p] = 1  # set corresponding columns to 1    
        dataframe = dataframe.set_index('Name') 
        dataframe = dataframe.drop(columns=['new_column', 'new_new_column'])
        clusterer = pickle.load(open('/Users/Jeylan/Documents/clbng_web/finalized_model.sav', 'rb'))
        cluster_labels = pickle.load(open('/Users/Jeylan/Documents/clbng_web/finalized_labels.sav', 'rb'))
        cluster = clusterer.predict(dataframe)
        clusterprint = df3.loc[cluster_labels==cluster]
        clusterprint_L1 = clusterprint[(clusterprint['Language']==str(language1).title())]
        clusterprint_L1_g = clusterprint_L1[(clusterprint_L1['Gender']==str(gender).title())]
        clusterprint_L2 = clusterprint[(clusterprint['Language']==str(language2).title())]
        clusterprint_L2_g = clusterprint_L2[(clusterprint_L2['Gender']==str(gender).title())]
        rs = RefinedSoundex()
        sharedlist = list(set(lang_dict[language1]) & set(lang_dict[language2]))
        ## This name is not pronounceable in both languages. 
        st.write("This name is not pronounceable in both languages.")  
        
        ## 1 Here are similar names that are spelled and pronounced the same in both languages
        clusterprint_L1_g = clusterprint_L1_g.reset_index()
        NewList1 = [clusterprint_L1_g['Name'].iloc[i] for i,nc_val in enumerate(clusterprint_L1_g['new_column']) if set(nc_val).issubset(sharedlist)]
        clusterprint_L2_g = clusterprint_L2_g.reset_index()
        NewList2 = [clusterprint_L2_g['Name'].iloc[i] for i,nc_val in enumerate(clusterprint_L2_g['new_column']) if set(nc_val).issubset(sharedlist)]
        clusterprint_L1_g= clusterprint_L1_g.drop_duplicates(subset=['Name'])
        clusterprint_L2_g= clusterprint_L2_g.drop_duplicates(subset=['Name'])
        clusterprint_L3_g = clusterprint_L1_g.append(clusterprint_L2_g)
        
        duplicate = clusterprint_L3_g[clusterprint_L3_g.duplicated(['Name', 'new_column'])] 
        NewList3 = duplicate['Name'].tolist()
        NewList4 = clusterprint_L2_g['Name'].tolist()
        
        ##NewFrame5 = df3['new_column'][df3['Language2'] == language2]
        
        
        if len(NewList3) > 0:
        	st.write("Here are similar names that are spelled and pronounced the same in both languages:")
        	st.markdown(NewList3)
           	
        ## 2 Here are simialr names in English that are pronounceable in Spanish 
        if len(NewList1) > 0 and len(NewList1) < 40:
        	not_in_List3 = list(set(NewList1) - set(NewList3))
	        st.write("Here are similar names in " + str(language1).title() + " that are pronounceable in " + str(language2).title() + ":")
	        st.markdown(not_in_List3)

        if len(NewList1) >= 40 :
        	not_in_List3 = list(set(NewList1) - set(NewList3))
        	distances = [rs.distance(namez, str(firstname)) for namez in not_in_List3]
        	indexes = np.argsort(distances)
        	least_names = np.array(not_in_List3)[indexes]
	        st.write("Here are similar names in " + str(language1).title() + " that are pronounceable in " + str(language2).title() + ":")
        	st.markdown(least_names[:40].tolist())

        if len(NewList2)< 40 and len(NewList2) > 0:
        	not_in_List3 = list(set(NewList2) - set(NewList3))
	        st.write("Here are similar names in " + str(language2).title() + " that are pronounceable in " + str(language1).title() + ":")
	        st.markdown(not_in_List3)

        if len(NewList2)>=40:
        	not_in_List3 = list(set(NewList2) - set(NewList3))
        	distances = [rs.distance(namez, str(firstname)) for namez in not_in_List3]
        	indexes = np.argsort(distances)
        	least_names = np.array(not_in_List3)[indexes]
	        st.write("Here are similar names in " + str(language2).title() + " that are pronounceable in " + str(language1).title() + ":")
        	st.markdown(least_names[:40].tolist())         
        
        
        if firstname in NewList4:
        	simname = clusterprint_L2_g['new_column'][clusterprint_L2_g['Name'] == firstname]
        	st.write("While this name is not pronounceable in " + str(language2) + ", the name '" +  str(firstname) + "' exists with the same spelling but a different pronunciation in " + str(language2) + ": " + str(simname.tolist()))

               
        if len(NewList3) == 0 and len(NewList1) == 0 and len(NewList2) == 0 and firstname not in NewList4: 
        	st.write("Sorry, we cannot find any suggestions.")
        
        st.write(clusterprint_L1_g)
        st.write(clusterprint_L2_g)
        st.markdown(sharedlist)
        st.write(NewList2)

#st.write("fourth part")

#end = timer()
#st.write(end - start) 


#start = timer()


if st.button("Submit"): function(firstname, language1, language2, gender)
	




##st.write("fifth part")


#end = timer()
#st.write(end - start) 


