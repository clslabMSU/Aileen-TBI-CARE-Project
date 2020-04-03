'''
author: Aileen
purpose:
Get the data for Sujit. Referencing the file 05_trajectory_data.py

Created: 6/3/2020
Last Modified: 27/3/2020


'''

import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime
import time

# ************** Pre processing Data *******************
def fixContext(context): #To fix some typo in the original data files
    context = str(context)
    context = context.strip()
    if context == "6 m onths post-injury":
        return "6 months post-injury"
    elif context.endswith(" 1") or context.endswith(" 2") or context.endswith(" 3"):
        return context[:-2]
    elif context.endswith("1") or context.endswith("2") or context.endswith("3"):
        if "Unrestricted to Play" in context:
            return "Unrestricted Return to Play"
        else:
            return context[:-1]
    elif context == 'nan':
        return "Baseline"
    elif "Unrestricted to Pla" in context:
        return "Unrestricted Return to Play"
    else:
        return context

def get_input(path,master_list,cols): #To get the data of people in the master_list only
    df = pd.read_csv(path, low_memory=False)
    real_file_name = path.split(sep="result")[-1].split(sep="2019")[0]

    for col in df.columns:
        if col.endswith("GUID"):
            guid_col = col
            cols.insert(0, col)
        elif col.endswith("VisitDate"):
            date_col = col
            cols.insert(1, col)
        elif col.endswith("ContextTypeOTH"):
            con_col = col
            cols.insert(2, col)
        # elif col.endswith("AgeYrs"):
        #     age_col = col
        #     cols.insert(3, col)  
    try:
        temp = df[[guid_col, date_col, con_col]]
    except:
        for col in df.columns: 
            if col.endswith("GeneralNotesTxt"):
                con_col = col
                cols.insert(2, col)
        try:
            temp = df[[guid_col, date_col, con_col]]  
        except:
            print("MISSING ONE OF THE COLUMNS", real_file_name)
            exit()

    df = df[cols]
    df.rename(columns = {guid_col: 'GUID', con_col: 'Context'}, inplace = True)
    df = pd.merge(df, master_list, on = 'GUID', how = 'inner')

    df.insert(3,'Remark','')
    
    #****** Applies Cleaning function for readability *******************
    df[date_col] = pd.to_datetime(df[date_col])
    df['Context'] = df['Context'].apply(fixContext)

    df = df.sort_values(by = ['GUID', date_col], ascending = [True, True]) #O(nlogn)


    return df

# ************** Select data *******************
def status_update(stt, guid, s1, s2, s3, s4, s5, s6,s7):
    if s1 == "":
        s1 = stt[guid]['Baseline']
    if s2 == "":
        s2 = stt[guid]['< 6 hours']
    if s3 == "":
        s3 = stt[guid]['24-48 hours']
    if s4 == "":
        s4 = stt[guid]['Asymptomatic']
    if s5 == "":
        s5 = stt[guid]['Unrestricted Return to Play']
    if s6 == "":
        s6 = stt[guid]['7 days Post-Unrestricted Return to Play']
    if s7 == "":
        s7 = stt[guid]['6 months post-injury']

    stt.update({guid:{'Baseline': s1, '< 6 hours': s2, '24-48 hours': s3, 'Asymptomatic': s4, 'Unrestricted Return to Play': s5,\
        '7 days Post-Unrestricted Return to Play': s6, '6 months post-injury': s7}})

    return stt
    
def select_trajectory(df):
    
    #---- Loop through the data to get the 1st concussion date & repeated concussion for each GUID
    concussion = dict.fromkeys(df.GUID.unique(),{'1st concussion': '', '2nd concussion': ''})

    # Check for 1st concussion date
    for row in range(len(df)): #O(n)
        guid, date, context = df.iloc[row,0], df.iloc[row,1], df.iloc[row,2]
        if context == '< 6 hours' or context == '24-48 hours':
            if concussion[guid]['1st concussion'] == '':
                concussion.update({guid:{'1st concussion' : date, '2nd concussion' : ''}})

    # Check for repeated (2nd) concussion date
    for row in range(len(df)): #O(n)
        guid, date, context = df.iloc[row,0], df.iloc[row,1], df.iloc[row,2]      
        if concussion[guid]['1st concussion'] != '': #only check those who had 1st injury (just for code performance purpose)
            if context == '< 6 hours' or context == '24-48 hours':
                if concussion[guid]['2nd concussion'] == '':
                    dist = date - concussion[guid]['1st concussion']
                    if dist.days > 10: #only accept repeat concussion if it happens after 10 days
                        concussion.update({ guid:{'1st concussion' : concussion[guid]['1st concussion'],'2nd concussion' : date}})

    #---- Loop through the df again and get the trajectory, mark selected/skip
    status = dict.fromkeys(df.GUID.unique(),{'Baseline': '', '< 6 hours': '', '24-48 hours': '', 'Asymptomatic': '', \
        'Unrestricted Return to Play': '','7 days Post-Unrestricted Return to Play': '', '6 months post-injury': ''})

    for row in range(len(df)):
        guid, date, context = df.iloc[row,0], df.iloc[row,1], df.iloc[row,2]
        if concussion[guid]['1st concussion'] != '':
            dist_1st = date - concussion[guid]['1st concussion']
            dist_1st = dist_1st.days
        if concussion[guid]['2nd concussion'] != '':
            dist_2nd = date - concussion[guid]['2nd concussion']
            dist_2nd = dist_2nd.days

        if context == 'Baseline': #take latest baseline before the first injury
            if concussion[guid]['1st concussion'] == '' or dist_1st < 0: 
                status = status_update(status,guid,date,'','','','','','')
        
        if context == '< 6 hours' and status[guid]['< 6 hours'] == '': #take the first <6 hours
            status = status_update(status,guid,'',date,'','','','','')

        if context == '24-48 hours' and status[guid]['24-48 hours'] == '': #take the first 24-48 hours
            status = status_update(status,guid,'','',date,'','','','')
        
        if context == 'Asymptomatic': #take the last data point before any repeated injury
            if concussion[guid]['2nd concussion'] == '' or dist_2nd < 0: 
                status = status_update(status,guid,'','','',date,'','','')
        
        if context == 'Unrestricted Return to Play': #take the last data point before any repeated injury
            if concussion[guid]['2nd concussion'] == '' or dist_2nd < 0: 
                status = status_update(status,guid,'','','','',date,'','')

        if context == '7 days Post-Unrestricted Return to Play': #take the last data point before any repeated injury
            if concussion[guid]['2nd concussion'] == '' or dist_2nd < 0: 
                status = status_update(status,guid,'','','','','',date,'')

        if context == '6 months post-injury': #take the last data point before any repeated injury
            if concussion[guid]['2nd concussion'] == '' or dist_2nd < 0:
                status = status_update(status,guid,'','','','','','',date)

    return status

def final_trajectory(df, status, sample_list):    
    # Get the data only for the selected trajectories (drop the repeated onjuries data)
    for row in range(len(df)):
        guid, date, context = df.iloc[row,0], df.iloc[row,1], df.iloc[row,2]
        if status[guid][context] == date:
            df.iloc[row,3] = 'Selected'
        else:
            df.iloc[row,3] = 'Skip'

    df = df[df['Remark'] == 'Selected']
    df.iloc[:,1] = df.iloc[:,1].apply(lambda x: str(x)[0:10] if 'time' in str(type(x)) else x)
    df = pd.merge(sample_list, df, on = ['GUID', 'Gender', 'CC']).drop(columns = ['Remark', df.columns[1]])
    
    return df

def output_trajectory(status):
    out = pd.DataFrame(status).T
    out = out[['Baseline','< 6 hours','24-48 hours','Asymptomatic','Unrestricted Return to Play','7 days Post-Unrestricted Return to Play','6 months post-injury']]
    
    for c in out.columns: # Format the timestamp to get the date for output 
        out[c] = out[c].apply(lambda x: str(x)[0:10] if 'time' in str(type(x)) else x)  
        # out[c] = out[c].map(lambda x: datetime.strptime(str(x)[:10], '%Y-%m-%d') if 'Timestamp' in str(type(x)) else x)
    out = out.reset_index().rename(columns = {'index': 'GUID'})

    return out

def main():
    start_time = time.time()

    #*********** Loop through the selected test and get the trajectory ************************ 
    test_list = \
    {'_SAC_': ['SAC.Scoring Summary.SACOrientationSubsetScore','SAC.Scoring Summary.SACImmdMemorySubsetScore','SAC.Scoring Summary.SACConcentationSubsetScore','SAC.Scoring Summary.SACDelayedRecallSubsetScore','SAC.Scoring Summary.SACTotalScore'],
    '_BESS_': ['BESS.Balance Error Scoring Test.BESSTotalFirmErrorCt','BESS.Balance Error Scoring Test.BESSTotalFoamErrorCt','BESS.Balance Error Scoring Test.BESSTotalErrorCt'],
    '_SCAT3_': ['SCAT3.Scoring Summary.Scat3TotalSymptoms','SCAT3.Scoring Summary.Scat3TotSympScore'],
    '_BSI18_': ['BSI18.Form Completion.BSI18SomScoreRaw','BSI18.Form Completion.BSI18DeprScoreRaw','BSI18.Form Completion.BSI18AnxScoreRaw','BSI18.Form Completion.BSI18GSIScoreRaw'],
    '_ImPACT_': ['ImPACT.Post-Concussion Symptom Scale (PCSS).ImPACTTotalSymptomScore','ImPACT.ImPACT Test.ImPACTVerbMemoryCompScore','ImPACT.ImPACT Test.ImPACTVisMemoryCompScore','ImPACT.ImPACT Test.ImPACTVisMotSpeedCompScore','ImPACT.ImPACT Test.ImPACTReactTimeCompScore','ImPACT.ImPACT Test.ImPACTImplseCntrlCompScore']}

    naming = {'SAC.Scoring Summary.SACOrientationSubsetScore': 'SAC Orientation Score',
    'SAC.Scoring Summary.SACImmdMemorySubsetScore': 'SAC Immediate Memory Score',
    'SAC.Scoring Summary.SACConcentationSubsetScore': 'SAC Concentration Score',
    'SAC.Scoring Summary.SACDelayedRecallSubsetScore': 'SAC Delayed Recall Score',
    'SAC.Scoring Summary.SACTotalScore': 'SAC Overall Score',
    'BESS.Balance Error Scoring Test.BESSTotalFirmErrorCt': 'BESS Total Firm Error',
    'BESS.Balance Error Scoring Test.BESSTotalFoamErrorCt': 'BESS Total Foam Error',
    'BESS.Balance Error Scoring Test.BESSTotalErrorCt': 'Total Error',
    'SCAT3.Scoring Summary.Scat3TotalSymptoms': 'SCAT3 Total Symptoms',
    'SCAT3.Scoring Summary.Scat3TotSympScore': 'SCAT3 Total Symptoms Score',
    'BSI18.Form Completion.BSI18SomScoreRaw': 'BSI18 Somatization Score',
    'BSI18.Form Completion.BSI18DeprScoreRaw': 'BSI18 Depression Score',
    'BSI18.Form Completion.BSI18AnxScoreRaw': 'BSI18 Anxiety Score',
    'BSI18.Form Completion.BSI18GSIScoreRaw': 'BSI18 GSI score',
    'ImPACT.Post-Concussion Symptom Scale (PCSS).ImPACTTotalSymptomScore': 'ImPACT Total Symptom Score',
    'ImPACT.ImPACT Test.ImPACTVerbMemoryCompScore': 'ImPACT Verbal Memory Score',
    'ImPACT.ImPACT Test.ImPACTVisMemoryCompScore': 'ImPACT Visual Memory Score',
    'ImPACT.ImPACT Test.ImPACTVisMotSpeedCompScore': 'ImPACT Visual Motor Speed Score',
    'ImPACT.ImPACT Test.ImPACTReactTimeCompScore': 'ImPACT React Time Score',
    'ImPACT.ImPACT Test.ImPACTImplseCntrlCompScore': 'ImPACT Impulse Control Score'}

    #----------- Demographic ------------
    demo = pd.read_csv('/Users/aileenbui/Downloads/CARE/CARE dataset_2019-08-22T11-40-02/query_result_DemogrFITBIR_2019-08-22T11-33-593433479263719920494.csv', low_memory = False)\
        .rename(columns = {'DemogrFITBIR.Main Group.GUID': 'GUID', 'DemogrFITBIR.Subject Demographics.GenderTyp': 'Gender', 'DemogrFITBIR.Main Group.AgeYrs': 'Age'})
            # 'DemogrFITBIR.Subject Demographics.EthnUSACat': 'EthnUSACat'})
    demo = demo[['GUID', 'Gender']]
    print('Get Demographic Data')
    #----------- Post Injury ------------
    post_inj = pd.read_csv('/Users/aileenbui/Downloads/CARE/CARE dataset_2019-08-22T11-40-02/query_result_PostInjForm_2019-08-22T11-34-204240316875745846702.csv', low_memory = False)\
        .rename(columns = {'PostInjForm.Main Group.GUID': 'GUID', 'PostInjForm.Main Group.CaseContrlInd': 'CC',\
            'PostInjForm.Post Injury Description.LOCDur': 'LOCDur', 'PostInjForm.Post Injury Description.ConcussEvntType': 'ConcussEvntType',\
                'PostInjForm.Post Injury Description.ARCAthleteTyp':'ARCAthleteTyp', 'PostInjForm.Post Injury Description.LOCInd': 'LOCInd',\
                    'PostInjForm.Post Injury Description.TBIHospitalizedInd': 'TBIHospitalizedInd', 'PostInjForm.Post Injury Description.PstTraumtcAmnsInd': 'PTAInd',\
                        'PostInjForm.Main Group.VisitDate':'Visit_Date_Post_Inj'})
    post_inj = post_inj[['GUID', 'CC', 'Visit_Date_Post_Inj', 'LOCInd', 'LOCDur', 'PTAInd', 'ConcussEvntType', 'ARCAthleteTyp', 'TBIHospitalizedInd']]
    print('Get Post Injury Data')
    
    full = pd.merge(demo, post_inj, on = 'GUID')
    full = full[['GUID','Gender','CC']].drop_duplicates('GUID').dropna() #For Sujit
    master_result = full.copy()
    print('Full length', len(master_result))


    # inputFolder = "\\\\EGR-1L11QD2\\CLS_lab\\TBI data\\CARE data\\CARE dataset_2019-08-22T11-40-02\\" #work from lab
    inputFolder = "/Users/aileenbui/Downloads/CARE/CARE dataset_2019-08-22T11-40-02/" #work from home
    inputFiles = glob(inputFolder + "*.csv")
    # outputFolder = "\\\\EGR-1L11QD2\\CLS_lab\\Aileen\\TBI Support work\\CARE\\"

    with pd.ExcelWriter('Assessment_scores_both_case_control.xlsx') as w:
        for f in inputFiles: #Glob through the input folder
            file_name = f.split(sep="result")[-1].split(sep="2019")[0]
            if file_name in test_list.keys():
                print('Working on file ',file_name)
                data = get_input(f, full, test_list[file_name])
                selection = select_trajectory(data)
                trajectory_dict = output_trajectory(selection)
                # trajectory_dict = pd.merge(full, trajectory_dict, on = 'GUID')
                result = final_trajectory(data, selection, full)
                try:
                    master_result = pd.merge(master_result, result, on = ['GUID', 'Gender', 'CC', 'Context'], how = 'outer')
                except:
                    master_result = pd.merge(master_result, result, on = ['GUID', 'Gender', 'CC'], how = 'outer')
 
        for col in master_result.columns:
            if col in naming.keys():
                master_result.rename(columns = {col: naming[col]}, inplace = True)
        out1 = master_result[master_result['Context'] == 'Baseline']
        out1.to_excel(w, sheet_name = 'Baseline', engine='xlsxwriter', index = False)

        out2 = master_result[master_result['Context'] == '< 6 hours']
        out2.to_excel(w, sheet_name = '6 hours', engine='xlsxwriter', index = False)

        out3 = master_result[master_result['Context'] == '24-48 hours']
        out3.to_excel(w, sheet_name = '24-48 hours', engine='xlsxwriter', index = False)

    end_time = time.time()
    print('Elapsed time:', end_time - start_time)
    print('Elapsed time:', end_time - start_time)

main()
    