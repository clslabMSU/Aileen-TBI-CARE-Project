'''
author: Aileen
purpose:
Get the list of people who have scores accross the selected assessments.
Remove the duplicates in trajectory. Rules:
Take the first time they got <6 hours post injury record. (Event #2)
The baseline date is the last time they got record prior to the Event#2. (Event #1)
Take the first time they got 24-48 hours post injury record, after the event #2. (Event #3)
Take the last time they are marked asymptonic after event #3 and before any repeated injury. (Event #4)
Take the last time they are marked 7 days RTP after event #3 and before any repeated injury. (Event #5)
Take the last time they are marked 6 months RTP after event #3 and before any repeated injury. (Event #6)

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
    try:
        temp = df[[guid_col,date_col,con_col]]
    except:
        for col in df.columns: 
            if col.endswith("GeneralNotesTxt"):
                con_col = col
                cols.insert(2, col)
        try:
            temp = df[[guid_col,date_col,con_col]]  
        except:
            print("MISSING ONE OF THE COLUMNS", real_file_name)
            exit()

    df = df[cols]
    df.rename(columns = {guid_col: 'GUID', con_col: 'Context'}, inplace = True)
    df = pd.merge(df, master_list, on = 'GUID', how = 'inner')

    df.insert(3,'Remark','')
    
    #****** Applies Cleaning function for readability *******************
    # df[date_col] = df[date_col].apply(str)
    # df[date_col] = df[date_col].apply(lambda x: x.split(sep="T")[0])
    df[date_col] = pd.to_datetime(df[date_col])
  
    # df[con_col] = df[con_col].apply(str)
    df['Context'] = df['Context'].apply(fixContext)
    # print(real_file_name,df[con_col].unique())

    df = df.sort_values(by = ['GUID', date_col], ascending = [True, True]) #O(nlogn)

    return df

# ************** Functions *******************
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
    
def select_trajectory(df): #Apply the rules to select the first trajectory timepoints for each GUID
    
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

    ## -- This section is just to write to excel for testing process
    # cc = pd.DataFrame(concussion)
    # cc = cc.T
    # cc.to_csv('concuss.csv')

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
        
        if context == '< 6 hours' and status[guid]['< 6 hours'] == '': #take the 1st data point before any repeated injury
            if concussion[guid]['2nd concussion'] == '' or dist_2nd < 0:
                status = status_update(status,guid,'',date,'','','','','')

        if context == '24-48 hours' and status[guid]['24-48 hours'] == '': #take the 1st data point before any repeated injury
            if concussion[guid]['2nd concussion'] == '' or dist_2nd < 0:
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

def final_trajectory(df, status, sample_list): #Get the scores of the selected trajectories only   
    # Get the data only for the selected trajectories (drop the repeated onjuries data)
    for row in range(len(df)):
        guid, date, context = df.iloc[row,0], df.iloc[row,1], df.iloc[row,2]
        if status[guid][context] == date:
            df.iloc[row,3] = 'Selected'
        else:
            df.iloc[row,3] = 'Skip'

    df = df[df['Remark'] == 'Selected']
    df.iloc[:,1] = df.iloc[:,1].apply(lambda x: str(x)[0:10] if 'time' in str(type(x)) else x)
    df = pd.merge(sample_list, df, on = ['GUID', 'Gender', 'CC']).drop(columns = ['Remark',df.columns[1]])
    
    return df

def select_sample(status,file_name): #Select the sample without missing data at any timepoints upto RTP
    out = pd.DataFrame(status).T
    if file_name == '_SCAT3_':
        sampleList = set(out.index.tolist())
        return out, sampleList
    out = out[ out['Baseline'] != '' ]
    out = out[ out['24-48 hours'] != '' ]
    out = out[ out['Asymptomatic'] != '' ]
    out = out[ out['Unrestricted Return to Play'] != '' ]
    if file_name == '_SAC_' or file_name == '_BESS_': #Only these two tests have data at <6 hours
        out = out[ out['< 6 hours'] != '' ]

    sampleList = set(out.index.tolist())
    return out, sampleList

def output_trajectory(status): #Format and output the list of trajectories per each GUID
    out = pd.DataFrame(status).T
    out = out[['Baseline','< 6 hours','24-48 hours','Asymptomatic','Unrestricted Return to Play','7 days Post-Unrestricted Return to Play','6 months post-injury']]
    
    for c in out.columns: # Format the timestamp to get the date for output 
        out[c] = out[c].apply(lambda x: str(x)[0:10] if 'time' in str(type(x)) else x)  
        # out[c] = out[c].map(lambda x: datetime.strptime(str(x)[:10], '%Y-%m-%d') if 'Timestamp' in str(type(x)) else x)
    out = out.reset_index().rename(columns = {'index': 'GUID'})

    return out

def summary_trajetory(df,file_name): #Count the missing data per GUID
    df.insert(10,'continuous','')
    for row in range(len(df)): #Convert into number so we can use isnull() function for summary
        continuous = 0
        count = 0
        for i in range (3,10):
            if df.iloc[row,i] == '':
                df.iloc[row,i] = np.nan
                continuous += 1
                count +=1
            elif df.iloc[row,i] != '':
                df.iloc[row,i] = 1
                continuous = 0
        df.iloc[row,10] = continuous
    
    ### Get the count of missing data per GUID
    df['NAcount'] = df.apply(lambda x: x.isnull().sum(), axis=1)
    ### Get the count of missing data per feature
    df.loc[file_name] = df.isnull().sum()
    
    out = df.iloc[-1:].drop(columns = ['continuous','NAcount']) #Get the last row

    out['0 missing timepoints'] = (df.NAcount.values == 0).sum()
    out['1 missing timepoints'] = (df.NAcount.values == 1).sum()
    out['2 missing timepoints'] = (df.NAcount.values == 2).sum()
    out['3 missing timepoints'] = (df.NAcount.values == 3).sum()
    out['4 missing timepoints'] = (df.NAcount.values == 4).sum()
    out['5 missing timepoints'] = (df.NAcount.values == 5).sum()
    out['6 missing timepoints'] = (df.NAcount.values == 6).sum()
    out['7 missing timepoints'] = (df.NAcount.values == 7).sum()

    out['2 cont. missing timepoints'] = (df.continuous.values == 2).sum()
    out['3 cont. missing timepoints'] = (df.continuous.values == 3).sum()
    out['4 cont. missing timepoints'] = (df.continuous.values == 4).sum()
    out['5 cont. missing timepoints'] = (df.continuous.values == 5).sum()
    out['6 cont. missing timepoints'] = (df.continuous.values == 6).sum()
    out['7 cont. missing timepoints'] = (df.continuous.values == 7).sum()

    return out

def missing_data(df,file_name): # Count the missing data per each feature
    df['NAcount'] = df.apply(lambda x: x.isnull().sum(), axis=1)
    out1 = df.groupby(['Context','NAcount']).size()
    df.loc[file_name] = df.isnull().sum()
    out2 = df.iloc[-1:]

    return out1, out2

def merge_df(df1,df2,df3,df4,df5,df6):
    out = pd.merge(df1, df2, on = ['GUID', 'Gender'], how = 'left')
    out = pd.merge(out, df3, on = ['GUID', 'CC'], how = 'left')
    out = pd.merge(out, df4, on = 'GUID', how = 'left')
    out = pd.merge(out, df5, on = 'GUID', how = 'left')
    out = pd.merge(out, df6, on = 'GUID', how = 'left')

    return out

def dup_feature(df1,df2,W,file):
    if file == 'post_inj':
        out = pd.merge(df1, df2, on = ['GUID', 'CC'], how = 'left')
    else:
        out = pd.merge(df1, df2, on = ['GUID'], how = 'left')
    out = out[out.duplicated(subset = 'GUID', keep = False)]
    out.sort_values(by = ['GUID'], inplace = True)
    out.to_excel(W, sheet_name = file, engine = 'xlsxwriter', index = False)

def main():
    start_time = time.time()

    #*********** A. Get the selected features from original data ***********************
    #----------- A1. Demographic ------------
    demo = pd.read_csv('/Users/aileenbui/Downloads/CARE/CARE dataset_2019-08-22T11-40-02/query_result_DemogrFITBIR_2019-08-22T11-33-593433479263719920494.csv', low_memory = False)\
        .rename(columns = {'DemogrFITBIR.Main Group.GUID': 'GUID', 'DemogrFITBIR.Subject Demographics.GenderTyp': 'Gender', 'DemogrFITBIR.Subject Demographics.EthnUSACat': 'EthnUSACat'})
    demo = demo[['GUID', 'Gender', 'EthnUSACat']]
    print('Get Demographic Data')
    #----------- A2. Post Injury ------------
    post_inj = pd.read_csv('/Users/aileenbui/Downloads/CARE/CARE dataset_2019-08-22T11-40-02/query_result_PostInjForm_2019-08-22T11-34-204240316875745846702.csv', low_memory = False)\
        .rename(columns = {'PostInjForm.Main Group.GUID': 'GUID', 'PostInjForm.Main Group.CaseContrlInd': 'CC',\
            'PostInjForm.Post Injury Description.LOCDur': 'LOCDur', 'PostInjForm.Post Injury Description.ConcussEvntType': 'ConcussEvntType',\
                'PostInjForm.Post Injury Description.ARCAthleteTyp':'ARCAthleteTyp', 'PostInjForm.Post Injury Description.LOCInd': 'LOCInd',\
                    'PostInjForm.Post Injury Description.TBIHospitalizedInd': 'TBIHospitalizedInd', 'PostInjForm.Post Injury Description.PstTraumtcAmnsInd': 'PTAInd',\
                        'PostInjForm.Post Injury Description.PstTraumaticAmnsDur': 'PTADur', 'PostInjForm.Main Group.VisitDate':'Visit_Date_Post_Inj'})
    post_inj = post_inj[['GUID', 'CC', 'Visit_Date_Post_Inj', 'LOCInd', 'LOCDur', 'PTAInd', 'PTADur', 'ConcussEvntType', 'ARCAthleteTyp', 'TBIHospitalizedInd']]
    print('Get Post Injury Data')
    #----------- A3. Concussion History ------------
    concuss_hx = pd.read_csv('/Users/aileenbui/Downloads/CARE/CARE dataset_2019-08-22T11-40-02/query_result_Concussion_Hx_0000310_2_2019-08-22T11-36-002178571715763674339.csv', low_memory = False)\
        .rename(columns = {'Concussion_Hx_0000310_2.Main Group.GUID': 'GUID', #'Concussion_Hx_0000310_2.Form Administration.ContextTypeOTH': 'Concussion_Hx_Context',\
            'Concussion_Hx_0000310_2.Previous Concussion.HistHeadInjOrConcussInd': 'HistHeadInjOrConcussInd', 'Concussion_Hx_0000310_2.Previous Concussion.ConcussionPriorNum': 'PriorNumConcussion',\
                'Concussion_Hx_0000310_2.Previous Concussion.SportRelatedConcussionInd': 'SportRelatedConcussionInd', 'Concussion_Hx_0000310_2.Previous Concussion.DiagnosedInd': 'DiagnosedInd',\
                    'Concussion_Hx_0000310_2.Previous Concussion.AgeAtTBIEvent': 'AgeAtTBIEvent', 'Concussion_Hx_0000310_2.Previous Concussion.TBILocInd': 'TBILocInd',\
                        'Concussion_Hx_0000310_2.Previous Concussion.LOCDurRang': 'LOCDurRang', 'Concussion_Hx_0000310_2.Previous Concussion.LOCUnkDurInd': 'LOCUnkDurInd',\
                            'Concussion_Hx_0000310_2.Previous Concussion.AmnesiaConcussInd': 'AmnesiaConcussInd', 'Concussion_Hx_0000310_2.Previous Concussion.AmnesiaConcussDurRange': 'AmnesiaConcussDurRange',\
                                'Concussion_Hx_0000310_2.Previous Concussion.AmnsDurUnkInd': 'AmnsDurUnkInd', 'Concussion_Hx_0000310_2.Previous Concussion.ConcussionSymptomDurDays': 'ConcussionSymptomDurDays',\
                                    'Concussion_Hx_0000310_2.Previous Concussion.TBISxDurUnkInd': 'TBISxDurUnkInd',\
                                        'Concussion_Hx_0000310_2.Main Group.VisitDate': 'Visit_Date_Concuss_Hx'})
    concuss_hx = concuss_hx[['GUID', 'Visit_Date_Concuss_Hx', 'HistHeadInjOrConcussInd', 'PriorNumConcussion', 'SportRelatedConcussionInd', 'DiagnosedInd',\
        'AgeAtTBIEvent', 'TBILocInd', 'LOCDurRang', 'LOCUnkDurInd', 'AmnesiaConcussInd', 'ConcussionSymptomDurDays', 'AmnsDurUnkInd', 'ConcussionSymptomDurDays', 'TBISxDurUnkInd']]
    print('Get Concussion History Data')
    #----------- A4. Concussion History ------------
    demo_apx = pd.read_csv('/Users/aileenbui/Downloads/CARE/CARE dataset_2019-08-22T11-40-02/query_result_DemogrFITBIR_Appdx_0000310_2019-08-22T11-38-428884192043024417668.csv', low_memory = False)\
        .rename(columns = {'DemogrFITBIR_Appdx_0000310.Main.GUID': 'GUID', #'DemogrFITBIR_Appdx_0000310.Demographics.GenderTypExt': 'GenderTypExt',\
            'DemogrFITBIR_Appdx_0000310.Sport History.SportTeamParticipationTyp': 'SportTeamParticipationTyp', 'DemogrFITBIR_Appdx_0000310.Sport History.SportTeamParticipationTypOTH': 'SportTeamParticipationTypOTH',\
                'DemogrFITBIR_Appdx_0000310.Military Club Sport Team Participation.SportTeamParticipationTyp': 'SportTeamParticipationTyp',\
                    'DemogrFITBIR_Appdx_0000310.Main.VisitDate': 'Visit_Date_Demo_Appx'})
    demo_apx = demo_apx[['GUID', 'Visit_Date_Demo_Appx', 'SportTeamParticipationTyp', 'SportTeamParticipationTypOTH', 'SportTeamParticipationTyp']]
    print('Get Demographic Appendix Data')
    #----------- A5. Concussion History ------------
    med_hx = pd.read_csv('/Users/aileenbui/Downloads/CARE/CARE dataset_2019-08-22T11-40-02/query_result_MedHx_Appendix_CARE0000310_2019-08-22T11-39-136596062633385352422.csv', low_memory = False)\
        .rename(columns = {'MedHx_Appendix_CARE0000310.Main Group.GUID': 'GUID', #'MedHx_Appendix_CARE0000310.Form Adminstration.ContextTypeOTH': 'MedHx_context',\
            'MedHx_Appendix_CARE0000310.Medical History (You).HeadachesPastThreeMonthsInd': 'HeadachesPastThreeMonthsInd', 'MedHx_Appendix_CARE0000310.Medical History (You).HeadachWorkLimitAbilityInd': 'HeadachWorkLimitAbilityInd',\
                'MedHx_Appendix_CARE0000310.Medical History (You).HeadacheLightBotherInd': 'HeadacheLightBotherInd', 'MedHx_Appendix_CARE0000310.Medical History (You).HeadacheNasuseaInd': 'HeadacheNasuseaInd',\
                    'MedHx_Appendix_CARE0000310.Medical History (You).LearningDisordrDiagnosInd': 'LearningDisordrDiagnosInd',\
                        'MedHx_Appendix_CARE0000310.Medical History (You).ModerateSevereTBIDiagnosInd': 'ModerateSevereTBIDiagnosInd',\
                             'MedHx_Appendix_CARE0000310.Medical History (You).HeadachesDisordrDiagnosInd': 'HeadachesDisordrDiagnosInd',\
                                 'MedHx_Appendix_CARE0000310.Main Group.VisitDate': 'Visit_Date_Med_Hx'})
    med_hx = med_hx[['GUID', 'Visit_Date_Med_Hx', 'HeadachesPastThreeMonthsInd', 'HeadachWorkLimitAbilityInd', 'HeadacheLightBotherInd',  'HeadacheNasuseaInd', 'LearningDisordrDiagnosInd', 'ModerateSevereTBIDiagnosInd', 'HeadachesDisordrDiagnosInd']]                         
    print('Get Medical History Data')

    #*********** B. Loop through the selected test and get the trajectory ************************ 
    #----------- B1. List of selected assessments ------------
    test_list = \
    {'_SAC_': ['SAC.Scoring Summary.SACOrientationSubsetScore','SAC.Scoring Summary.SACImmdMemorySubsetScore','SAC.Scoring Summary.SACConcentationSubsetScore','SAC.Scoring Summary.SACDelayedRecallSubsetScore','SAC.Scoring Summary.SACTotalScore'],
    '_BESS_': ['BESS.Balance Error Scoring Test.BESSTotalFirmErrorCt','BESS.Balance Error Scoring Test.BESSTotalFoamErrorCt','BESS.Balance Error Scoring Test.BESSTotalErrorCt'],
    '_SCAT3_': ['SCAT3.Scoring Summary.Scat3TotalSymptoms','SCAT3.Scoring Summary.Scat3TotSympScore'],
    '_BSI18_': ['BSI18.Form Completion.BSI18SomScoreRaw','BSI18.Form Completion.BSI18DeprScoreRaw','BSI18.Form Completion.BSI18AnxScoreRaw','BSI18.Form Completion.BSI18GSIScoreRaw'],
    '_ImPACT_': ['ImPACT.Post-Concussion Symptom Scale (PCSS).ImPACTTotalSymptomScore','ImPACT.ImPACT Test.ImPACTVisMemoryCompScore','ImPACT.ImPACT Test.ImPACTVisMotSpeedCompScore','ImPACT.ImPACT Test.ImPACTReactTimeCompScore','ImPACT.ImPACT Test.ImPACTImplseCntrlCompScore']}
    
    #----------- B2. List of GUIDs who have tests result accross all the selected assessments ------------
    # master_case_list = pd.read_csv("\\\\EGR-1L11QD2\\CLS_lab\\Aileen\\TBI Support work\\CARE\\Output\\case_accross_assessments.csv", low_memory=False) #work from lab
    master_case_list = pd.read_csv("/Users/aileenbui/Downloads/CARE/case_accross_assessments.csv", low_memory = False) #work from home
    print('Get Master List')

    #----------- B3. Get the selected features ------------
    full = pd.merge(demo, post_inj, on = 'GUID')
    full = full[['GUID','Gender','CC']].drop_duplicates('GUID').dropna() #For Sujit
    print('Full length', len(full))

    final_sample = set(master_case_list.GUID.unique())
    # print('Final sample first starts at: ', len(final_sample))

    ##----------- B4. Specify the input folder for the glob function ------------
    # inputFolder = "\\\\EGR-1L11QD2\\CLS_lab\\TBI data\\CARE data\\CARE dataset_2019-08-22T11-40-02\\" #work from lab
    inputFolder = "/Users/aileenbui/Downloads/CARE/CARE dataset_2019-08-22T11-40-02/" #work from home
    inputFiles = glob(inputFolder + "*.csv")

    # #----------- B5. Get the sample list of people who do not have missing data in the trajectory (except 7 days & 6 months post RTP) ------------
    # for f in inputFiles: #Glob through the input folder
    #     file_name = f.split(sep="result")[-1].split(sep="2019")[0]
    #     if file_name in test_list.keys():
    #         print('Select sample: ',file_name)
    #         data = get_input(f, master_case_list, test_list[file_name])
    #         selection = select_trajectory(data)
    #         trajectory_dict, sample = select_sample(selection, file_name)
    #         final_sample = final_sample.intersection(sample)
    # print('Final sample: ', len(final_sample))
    # final_sample = full[full.GUID.isin(final_sample)]
    # final_sample.to_csv('Final_sample.csv', index = False)

    # exit()

    #-- After B5 section is run, comment it and run the code again.

    #----------- B6. Get the final trajectory and ouput ------------
    #----------- Create the holder for the output of summary ------------
    data_sum = pd.DataFrame(columns = ['Baseline','< 6 hours','24-48 hours','Asymptomatic','Unrestricted Return to Play','7 days Post-Unrestricted Return to Play','6 months post-injury',\
        '0 missing timepoints','1 missing timepoints','2 missing timepoints','3 missing timepoints','4 missing timepoints','5 missing timepoints','6 missing timepoints','7 missing timepoints',\
            '2 cont. missing timepoints','3 cont. missing timepoints','4 cont. missing timepoints','5 cont. missing timepoints','6 cont. missing timepoints','7 cont. missing timepoints'])

    final_sample = pd.read_csv('/Users/aileenbui/Downloads/CARE/Final_sample.csv', low_memory = False)
    master_result = final_sample.copy()

    with pd.ExcelWriter('Summary.xlsx') as w1:   
        with pd.ExcelWriter('Trajectory.xlsx') as w2:
            for f in inputFiles: #Glob through the input folder
                file_name = f.split(sep="result")[-1].split(sep="2019")[0]
                if file_name in test_list.keys():
                    print('Working on file ',file_name)
                    data = get_input(f, final_sample, test_list[file_name])
                    selection = select_trajectory(data)
                    trajectory_dict = output_trajectory(selection)
                    trajectory_dict = pd.merge(final_sample, trajectory_dict, on = 'GUID')
                    result = final_trajectory(data, selection, final_sample)
                    try:
                        master_result = pd.merge(master_result, result, on = ['GUID', 'Gender', 'CC', 'Context'], how = 'outer')
                    except:
                        master_result = pd.merge(master_result, result, on = ['GUID', 'Gender', 'CC'], how = 'outer')
                    trajectory_dict.to_excel(w2, sheet_name = file_name + '_trajectory', engine = 'xlsxwriter', index = False)
                    result.to_excel(w2, sheet_name = file_name + '_score', engine = 'xlsxwriter', index = False)

                    sum = summary_trajetory(trajectory_dict, file_name)
                    data_sum = pd.concat([data_sum, sum], sort = False)
                    out1, out2 = missing_data(result, file_name)
                    out1.to_excel(w1, sheet_name = file_name + 'countNA features', engine='xlsxwriter')
                    out2.to_excel(w1, sheet_name = file_name + 'countNA GUIDs', engine='xlsxwriter')

            data_sum.to_csv('Summary_missing_trajectory_final_sample.csv')
            master_result.to_csv('Assessment_scores.csv', index = False)
            feature = merge_df(final_sample, demo, post_inj, concuss_hx, demo_apx, med_hx)
            feature.to_csv('Selected_Features.csv', index = False)

            with pd.ExcelWriter('Duplicated_Features.xlsx') as W:
                dup_feature(final_sample, post_inj, W, 'post_inj')
                dup_feature(final_sample, concuss_hx, W, 'concuss_hx')
                dup_feature(final_sample, demo_apx, W, 'demo_apx')
                dup_feature(final_sample, med_hx, W, 'med_hx')

    end_time = time.time()
    print('Elapsed time:', end_time - start_time)
    print('Elapsed time:', end_time - start_time)

main()
    